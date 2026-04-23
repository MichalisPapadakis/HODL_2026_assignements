import itertools
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from challenge4 import EPOCHS, MazeGNN, SEED, device, evaluate_model, get_data


def _eval_fold_metrics(model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    model.eval()
    total_nodes = 0
    total_correct = 0
    total_graphs = 0
    total_perfect = 0

    with torch.no_grad():
        for batch in dataloader:
            n = len(batch.x)
            batch = batch.to(device)
            pred = model(batch, n)
            y_pred = torch.argmax(pred, dim=1)

            graph_correct = torch.sum(y_pred == batch.y).item()
            total_correct += graph_correct
            total_nodes += n
            total_graphs += 1
            if graph_correct == n:
                total_perfect += 1

    return {
        "node_accuracy": total_correct / max(1, total_nodes),
        "graph_accuracy": total_perfect / max(1, total_graphs),
    }


def _train_for_epochs(
    model: torch.nn.Module,
    train_data: List,
    epochs: int,
    lr: float = 4e-4,
) -> torch.nn.Module:
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, batch.num_nodes)
            loss = criterion(pred, batch.y.to(torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def _kfold_indices(n_samples: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop
    return folds


def run_5fold_cv_search():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_dataset, _ = get_data()
    n_samples = len(train_dataset)
    folds = _kfold_indices(n_samples=n_samples, k=5, seed=SEED)

    hidden_dims = [32, 64, 128]
    dropouts = [0.0, 0.1, 0.2, 0.3]
    aggregations = ["add", "mean", "max", "min"]

    search_space = list(itertools.product(hidden_dims, dropouts, aggregations))

    best_cfg = None
    best_score = float("-inf")

    print(f"Running 5-fold CV over {len(search_space)} configurations...")
    for hidden_dim, dropout, aggr in search_space:
        fold_graph_acc = []
        for train_idx, val_idx in folds:
            train_split = [train_dataset[i] for i in train_idx]
            val_split = [train_dataset[i] for i in val_idx]

            model = MazeGNN(hidden_dim=hidden_dim, dropout=dropout, aggr=aggr).to(device)
            model = _train_for_epochs(model, train_split, epochs=max(1, EPOCHS // 2))

            val_loader = DataLoader(val_split, batch_size=1, shuffle=False)
            metrics = _eval_fold_metrics(model, val_loader)
            fold_graph_acc.append(metrics["graph_accuracy"])

        mean_graph_acc = float(np.mean(fold_graph_acc))
        print(
            f"cfg(hidden_dim={hidden_dim}, dropout={dropout}, aggr={aggr}) "
            f"-> cv_graph_acc={mean_graph_acc:.4f}"
        )

        if mean_graph_acc > best_score:
            best_score = mean_graph_acc
            best_cfg = {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "aggr": aggr,
            }

    if best_cfg is None:
        raise RuntimeError("No valid configuration found during CV.")

    print("\nBest CV config:", best_cfg, f"with graph_accuracy={best_score:.4f}")
    print("Retraining best model on full training set...")

    best_model = MazeGNN(
        hidden_dim=best_cfg["hidden_dim"],
        dropout=best_cfg["dropout"],
        aggr=best_cfg["aggr"],
    ).to(device)
    best_model = _train_for_epochs(best_model, train_dataset, epochs=EPOCHS)

    print("\nDeploying best model to all grid sizes (4, 8, 16, 32):")
    results = evaluate_model(best_model, test_dataset=[])
    return best_model, best_cfg, results


if __name__ == "__main__":
    run_5fold_cv_search()
