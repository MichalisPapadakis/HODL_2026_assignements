import random
from copy import deepcopy
from pathlib import Path
from typing import Any

# ===== NN / Training Runtime Imports =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, MessagePassing, SAGEConv
from torch_geometric.utils import from_networkx, to_networkx
from math import ceil

# ===== Auxiliary / Plotting Imports =====
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DO_PLOTS = True
EPOCHS = 50
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)


### === PLOTTING SECTION === ###
def plot_path_predictions(
    model: torch.nn.Module,
    dataset,
    n_graphs: int = 20,
    out_dir: str = "generated_paths",
) -> None:
    if not DO_PLOTS:
        return

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    max_graphs = min(n_graphs, len(dataset))

    with torch.no_grad():
        for idx in range(max_graphs):
            data = dataset[idx]
            batch = data.to(device)
            pred_logits = model(batch, batch.num_nodes)
            pred_mask = torch.argmax(pred_logits, dim=1).cpu().numpy()

            nx_graph = to_networkx(data, to_undirected=True)
            nodelist = sorted(nx_graph.nodes())

            grid_size = int(getattr(data, "grid_size", int(np.sqrt(data.num_nodes))))
            pos = {
                node: (node % grid_size, -(node // grid_size))
                for node in nodelist
            }

            # Draw exactly the stored maze connectivity.
            draw_edges = list(nx_graph.edges())

            true_mask = data.y.cpu().numpy()
            true_colors = [
                "tab:orange" if int(true_mask[node]) == 1 else "lightgray"
                for node in nodelist
            ]
            pred_colors = [
                "tab:blue" if int(pred_mask[node]) == 1 else "lightgray"
                for node in nodelist
            ]

            plt.figure(figsize=(11, 5))
            plt.subplot(1, 2, 1)
            plt.title("True path (DFS / shortest path)")
            nx.draw(
                nx_graph,
                pos=pos,
                nodelist=nodelist,
                edgelist=draw_edges,
                node_color=true_colors,
                with_labels=False,
                node_size=70,
            )

            plt.subplot(1, 2, 2)
            plt.title("GNN prediction")
            nx.draw(
                nx_graph,
                pos=pos,
                nodelist=nodelist,
                edgelist=draw_edges,
                node_color=pred_colors,
                with_labels=False,
                node_size=70,
            )

            plt.tight_layout()
            plt.savefig(output_path / f"path_{idx:03d}.png")
            plt.close()


### === NN FUNCTIONS === ###
class MazeConv(MessagePassing):
    def __init__(self, hidden_dim: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear( hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        msg = self.propagate(edge_index, x=x)
        return self.norm(x + self.update_mlp(msg))

    def message(self, x_j, x_i):
        return self.message_mlp(torch.cat([x_i, x_j], dim=-1))


class MazeGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int =64, dropout: float = 0.2, aggr: str = "add"):
        super().__init__()
                
        self.dropout = dropout
        self.encoder = self.get_mlp(2, hidden_dim, 2*hidden_dim,last_relu=True)
        self.input_memory = self.get_mlp(4 * hidden_dim, 2*hidden_dim, 2*hidden_dim, last_relu=True)
        self.conv1 = MazeConv(2*hidden_dim, aggr=aggr)
        # self.conv2 = MazeConv(hidden_dim)
        # self.conv3 = MazeConv(hidden_dim)
        self.decoder = self.get_mlp(2*hidden_dim, hidden_dim, 2, last_relu=False)

    def forward(self, data, num_nodes):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        encoded_input = x

        for i in range(ceil(3 * np.sqrt(num_nodes))):
            x = self.input_memory(torch.cat([encoded_input, x], dim=-1))
            # x = self.conv3(self.conv2(self.conv1(x, edge_index),edge_index),edge_index)
            x = self.conv1(x,edge_index)

        x = self.decoder(x)
        return F.log_softmax(x, dim=1)

    def get_mlp(self, input_dim, hidden_dim, output_dim, last_relu=True):
        modules = [
            torch.nn.Linear(input_dim, int(hidden_dim)), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(self.dropout), 
            torch.nn.BatchNorm1d(hidden_dim),            
            torch.nn.Linear(int(hidden_dim), output_dim)]
        if last_relu:
            modules.append(torch.nn.ReLU())
        return torch.nn.Sequential(*modules)


### === TRAINING FUNCTIONS === ###
def eval_model(model, dataloader):
    model.eval()
    acc = 0
    tot_nodes = 0
    tot_graphs = 0
    perf = 0
    gpred = []
    gsol = []

    for batch in dataloader:
        n = len(batch.x)
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch, int(n))

        y_pred = torch.argmax(pred, dim=1)
        tot_nodes += len(batch.x)
        tot_graphs += batch.num_graphs

        graph_acc = torch.sum(y_pred == batch.y).item()
        acc += graph_acc
        gpred.extend([int(p.item()) for p in y_pred])
        gsol.extend([int(p.item()) for p in batch.y])

        if graph_acc == n:
            perf += 1

    gpred = torch.tensor(gpred)
    gsol = torch.tensor(gsol)
    f1score = f1_score(gpred, gsol)
    return f"node accuracy: {acc/tot_nodes:.3f} | node f1 score: {f1score:.3f} | graph accuracy: {perf/tot_graphs:.3}"


def _fit_model(model, dataset, epochs=20, lr=4e-4):
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_split = 0.2
    train_size = int((1 - val_split) * len(dataset))
    train_loader = DataLoader(dataset[:train_size], batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)

    model.train()
    best_model = deepcopy(model)
    best_score = None

    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            pred = model(data, data.num_nodes)
            loss = criterion(pred, data.y.to(torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        metrics = eval_model(model, val_loader)
        graph_val = float(metrics.split("graph accuracy: ")[-1])
        score = (-graph_val, running_loss)
        print(
            f"Epoch: {epoch + 1} "
            f"loss: {running_loss / max(1, len(train_loader.dataset)):.5f} | {metrics}"
        )

        if best_score is None or score < best_score:
            best_score = score
            best_model = deepcopy(model)
            print("Stored new best model:", score)

    return best_model


# Do not change function signature
def init_model() -> torch.nn.Module:
    model = MazeGNN().to(device)
    return model


# Do not change function signature
def train_model(model: torch.nn.Module, train_dataset: Dataset[Any]) -> torch.nn.Module:

    model = model.to(device)
    best_model = _fit_model(model, train_dataset, epochs=EPOCHS, lr=4e-4)
    plot_path_predictions(best_model, train_dataset, n_graphs=min(20, len(train_dataset)))
    return best_model


# ====================================================================================
# ====================================================================================
# ====================================================================================
### === LOCAL EVALUATION && GET DATA FUNCTION (CAN BE EXCLUDED FOR SUBMISSION) === ###
### === DATASET GENERATION === ###
def _build_maze_tree_graph(grid_size: int, seed: int):
    grid_graph = nx.grid_2d_graph(grid_size, grid_size)
    rng = random.Random(seed)

    def to_id(node):
        r, c = node
        return r * grid_size + c

    # Randomize edge weights, then take MST => random spanning tree on grid.
    for u, v in grid_graph.edges():
        grid_graph[u][v]["w"] = rng.random()
    nx_tree_2d = nx.minimum_spanning_tree(grid_graph, weight="w")

    # Safety checks: one connected tree spanning all nodes.
    if not nx.is_tree(nx_tree_2d):
        raise ValueError("Generated maze graph is not a valid spanning tree.")
    if nx.number_connected_components(nx_tree_2d) != 1:
        raise ValueError("Generated maze graph has disconnected components.")

    for u, v in nx_tree_2d.edges():
        if abs(u[0] - v[0]) + abs(u[1] - v[1]) != 1:
            raise ValueError("Invalid edge in maze tree: non-neighbor nodes connected.")

    relabel_map = {node: to_id(node) for node in nx_tree_2d.nodes()}
    nx_graph = nx.relabel_nodes(nx_tree_2d, relabel_map)
    num_nodes = grid_size * grid_size
    if num_nodes > 1:
        if min(dict(nx_graph.degree()).values()) < 1:
            raise ValueError("Generated maze graph has isolated nodes.")
        if nx_graph.number_of_edges() != num_nodes - 1:
            raise ValueError("Generated maze graph does not have N-1 edges.")

    torch_rng = torch.Generator().manual_seed(seed)
    endpoints = torch.randperm(num_nodes, generator=torch_rng)[:2].tolist()
    source, target = endpoints[0], endpoints[1]
    path_nodes = set(nx.shortest_path(nx_graph, source=source, target=target))

    # Input features: one-hot encoding of source and target nodes
    graph = from_networkx(nx_graph)
    graph.x = torch.zeros(num_nodes, 2, dtype=torch.float32)
    graph.x[source, 0] = 1.0
    graph.x[target, 1] = 1.0

    # Labels: 1 for nodes on the path, 0 otherwise
    graph.y = torch.zeros(num_nodes, dtype=torch.long)
    for node in path_nodes:
        graph.y[node] = 1

    graph.source = source
    graph.target = target
    graph.grid_size = grid_size
    graph.path_nodes = torch.tensor(sorted(path_nodes), dtype=torch.long)
    return graph


def train_dataset_gen(
    n_samples: int = 200,
    grid_size: int = 4,
    seed: int = 0,
):
    """
    Generates training mazes as random spanning trees over a grid graph.
    For each sample, build an n x n grid and sample a spanning tree.
    """
    return [_build_maze_tree_graph(grid_size, seed + i) for i in range(n_samples)]


def get_data():
    """
    Local fallback so run() works standalone.
    Grader environments may replace this with their own loader.
    """
    train_dataset = train_dataset_gen(n_samples=400, grid_size=4, seed=SEED)
    test_dataset = []
    return train_dataset, test_dataset


def _eval_model_metrics(model, dataloader):
    model.eval()
    node_acc = 0
    tot_nodes = 0
    tot_graphs = 0
    perf = 0
    gpred = []
    gsol = []

    for batch in dataloader:
        n = len(batch.x)
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch, int(n))

        y_pred = torch.argmax(pred, dim=1)
        tot_nodes += len(batch.x)
        tot_graphs += batch.num_graphs

        graph_acc = torch.sum(y_pred == batch.y).item()
        node_acc += graph_acc
        gpred.extend([int(p.item()) for p in y_pred])
        gsol.extend([int(p.item()) for p in batch.y])
        if graph_acc == n:
            perf += 1

    gpred = torch.tensor(gpred)
    gsol = torch.tensor(gsol)
    f1score = f1_score(gpred, gsol)
    return {
        "node_accuracy": node_acc / tot_nodes,
        "node_f1": f1score,
        "graph_accuracy": perf / tot_graphs,
    }


def evaluate_model(model: torch.nn.Module, test_dataset):
    del test_dataset
    size_buckets = [4, 8, 16, 32]
    results = {}

    for size in size_buckets:
        dataset = train_dataset_gen(n_samples=50, grid_size=size, seed=SEED + 1000 * size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        metrics = _eval_model_metrics(model, dataloader)
        results[size] = metrics

        plot_path_predictions(
            model,
            dataset,
            n_graphs=20,
            out_dir=f"generated_paths/{size}",
        )

    print("Evaluation on test-size categories:")
    for size in size_buckets:
        m = results[size]
        print(
            f"{size}x{size} -> "
            f"node accuracy: {m['node_accuracy']:.3f} | "
            f"node f1 score: {m['node_f1']:.3f} | "
            f"graph accuracy: {m['graph_accuracy']:.3f}"
        )

    return results


# This is what is being called by the grader
def run():
  random.seed(42)

  # Get datasets for training and testing
  train_dataset, test_dataset = get_data()

  # Initialize the model using student's init_model function
  model = init_model()

  # Train the model using student's train_model function
  model = train_model(model, train_dataset)

  # Evaluate the model on the test set
  model.eval()  # Set the model to evaluation mode
  score = evaluate_model(model, test_dataset)
  
  return score


if __name__ == "__main__":
    run()