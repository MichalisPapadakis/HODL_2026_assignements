import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm


# Do not change function signature
def init_model() -> nn.Module:
  # Your code here
  device = "cuda" if torch.cuda.is_available() else "cpu"
  class DeepNeuralNet(nn.Module):
    num_classes: int = 10
    input_dim: int = 784
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self,
                 lin_layers: list =[ 512,256,128,64]):
        super().__init__()
        self.layers = nn.ModuleList()

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # Layer 0
        self.l0   = nn.Linear(self.input_dim,lin_layers[0])
        self.bn0  = nn.BatchNorm1d(lin_layers[0])

        # Layer 1
        self.l1   = nn.Linear(lin_layers[0],lin_layers[1])
        self.bn1  = nn.BatchNorm1d(lin_layers[1])

        # Layer 2
        self.l2   = nn.Linear(lin_layers[1],lin_layers[2])
        self.bn2  = nn.BatchNorm1d(lin_layers[2])

        # Layer 3
        self.l3   = nn.Linear(lin_layers[2],lin_layers[3])
        self.bn3  = nn.BatchNorm1d(lin_layers[3])

        # Output Layer
        self.l_out = nn.Linear(lin_layers[3],self.num_classes)
        self.softmax = nn.Softmax(dim=1)

        layers_ = [ 
           self.l0,self.bn0,self.relu,self.dropout,
           self.l1,self.bn1,self.relu,
           self.l2,self.bn2,self.relu,self.dropout,
           self.l3,self.bn3,self.relu,
           self.l_out,self.softmax
        ]
        self.layers = nn.Sequential(*layers_)

    def forward(self, input):
      x = input
      return self.layers(x)
  
  model = DeepNeuralNet().to(device)
  # Input dimension: [B, 1, 28, 28]  (B = batch size, grayscale MNIST image)
  # Output dimension: [B, 10]        (digits 0–9)
  
  # Hint: you can flatten the image in your forward loop
  # def forward(self, x: torch.Tensor) -> torch.Tensor:
  #     x = x.flatten(1, 3)   # [B, 1, 28, 28] -> [B, 784]
  #     return ...
  
  return model
 

# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
  NUM_epochs = 25
  FREQ_epoch=1
  LOG_LEVEL=3
  batch_size = 64
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Your code
  train_dataloader = DataLoader(dev_dataset, batch_size=batch_size, pin_memory=True)
  optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
  loss_fn = nn.CrossEntropyLoss()

  def training_loop(
        dataloader: torch.utils.data.DataLoader,
        net: nn.Module,
        loss_fn: nn.Module,
        optimiser: torch.optim.Optimizer,
        verbosity: int=3,
        device = device):
    size = len(dataloader.dataset)
    last_print_point = 0
    current = 0

    acc_loss = 0
    acc_count = 0
    net.train()
    # for every slice (X, y) of the training dataset
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device) # torch.Tensor
        y = y.to(device)

        # perform a forward pass to compute the outputs of the net
        pred = net(X)

        # calculate the loss between the outputs of the net and the desired outputs
        loss_val = loss_fn(pred, y)
        acc_loss += loss_val.item()
        acc_count += 1

        # zero the gradients computed in the previous step
        optimiser.zero_grad()

        # calculate the gradients of the parameters of the net
        loss_val.backward()

        # use the gradients to update the weights of the network
        optimiser.step()

        # compute how many datapoints have already been used for training
        current = batch * len(X)

        # report on the training progress roughly every 10% of the progress
        if verbosity >= 3 and (current - last_print_point) / size >= 0.1:
            loss_val = loss_val.item()
            last_print_point = current
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    return acc_loss / acc_count

  def testing_loop(dataloader: DataLoader, net:nn.Module):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    net.eval()
    correct = 0
    

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return correct / size

  def train(dataloader, net, loss_fn, optimiser, epochs, epoch_frequency=50, device=device, verbosity=3):
    least_loss = None
    if verbosity < 2:
        for t in tqdm(range(epochs)):
            mean_loss = training_loop(dataloader, net, loss_fn, optimiser, verbosity=verbosity)
            accuracy = testing_loop(dataloader, net)
            if not least_loss or mean_loss < least_loss:
                least_loss = mean_loss
    else:
        for t in range(epochs):
            if verbosity >= 3:
                print(f"Epoch {t+1}\n-------------------------------")

            mean_loss = training_loop(dataloader, net, loss_fn, optimiser, verbosity=verbosity)
            accuracy = testing_loop(dataloader, net)
            if not least_loss or mean_loss < least_loss:
                least_loss = mean_loss

            if verbosity >= 2 and t%epoch_frequency == 0:
                print(f"Epoch {t:4}: mean loss {mean_loss:.5f}, validation accuracy {accuracy:7.2%}")
            if verbosity >= 3:
                print("\n")

    if verbosity >= 1:
        print(f"\nTraining complete, least loss {least_loss}, final validation accuracy {accuracy:.2%}")

    return least_loss

  train(train_dataloader, 
        model, 
		loss_fn, 
        optimiser, 
      	epochs=NUM_epochs, 
      	epoch_frequency=FREQ_epoch, 
      	verbosity=LOG_LEVEL)


  return model

from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda



def get_data():
    training_data = datasets.FashionMNIST(
        root="data_scratch",
        train=True,
        download=True,
        transform=Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x, start_dim=0))
        ]),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data_scratch",
        train=False,
        download=True,
        transform=Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x, start_dim=0))
        ]),
    )

    return training_data, test_data

def run():

  # Get datasets for training and testing
  train_dataset : Dataset
  test_dataset  : Dataset
  train_dataset, test_dataset = get_data()

  # Initialize the model using student's init_model function
  model = init_model()

  # Train the model using student's train_model function
  model = train_model(model, train_dataset)

  # Evaluate the model on the test set
  model.eval()  # Set the model to evaluation mode


  score = None
#   score = evaluate_model(model, test_dataset)
  
  return score

if __name__ == "__main__":
  run()