import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary # important to see network
from torch.nn.functional import relu, log_softmax


### Example Dataset
from torchvision.transforms import ToTensor

data_train = torchvision.datasets.MNIST("data_scratch",
        download=True, train=True, transform=ToTensor())
data_test = torchvision.datasets.MNIST("data_scratch",
        download=True, train=False, transform=ToTensor())

### Viewing Images and data statistics
fig,ax = plt.subplots(1,7)
for i in range(7):
    ax[i].imshow(data_train[i][0].view(28,28), cmap='gray')
    ax[i].set_title(data_train[i][1])
    ax[i].axis('off')

print('Training samples:', len(data_train))
print('Test samples:', len(data_test))
print('Tensor size:', data_train[0][0].size())
print('First 10 digits are:', [data_train[i][1] for i in range(10)])
print('Min intensity value: ', data_train[0][0].min().item())
print('Max intensity value: ', data_train[0][0].max().item())

#### =======================
### Helpers
### ========================
def plot_results(hist):
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['val_acc'], label='Validation acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()


#### =======================
### eXAMPLE NET DEFINTION
### ========================

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(784,100)
        self.out = nn.Linear(100,10)

        # or using Sequential
        # layers_ = [ 
        #    self.l0,self.bn0,self.relu,self.dropout,
        #    self.l1,self.bn1,self.relu,
        #    self.l2,self.bn2,self.relu,self.dropout,
        #    self.l3,self.bn3,self.relu,
        #    self.l_out
        # ]
        # self.function = nn.Sequential(*layers_)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = relu(x)
        x = self.out(x)
        x = log_softmax(x,dim=0)

        #using sequential
        # x = self.function(x)
        return x

#### =======================
### MAIN TRAINING CODE
### ========================

train_loader = torch.utils.data.DataLoader(data_train, batch_size=64)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64) 

def train_epoch(net, dataloader, device, lr=0.01, optimiser=None, loss_fn = nn.NLLLoss()):
    optimiser = optimiser or torch.optim.Adam(net.parameters(), lr=lr)
    # Ensure the network is in training mode
    net.train()
    total_loss, acc, count = 0,0,0
    for features,labels in dataloader:
        # Ensure features/images and labels are on the correct device
        features, labels = features.to(device), labels.to(device)
        # Predict outputs for features and compute the loss between predictions
        # and labels
        out = net(features)
        loss = loss_fn(out, labels)
        total_loss += loss

        # Zero gradients from earlier computations
        optimiser.zero_grad()
        # Compute gradients for weights
        loss.backward()
        # Do optimiser step
        optimiser.step()

        # Get predicted classes as the entry with the highest probability
        _, predicted = torch.max(out, 1)
        # Compute accuracy
        acc += (predicted == labels).sum().cpu()
        count += len(labels)
    return total_loss.item()/count, acc.item()/count

def validate_epoch(net, dataloader, device, loss_fn=nn.NLLLoss()):
    # Ensure network is in evaluation mode
    net.eval()
    count,acc,loss = 0,0,0
    # We use torch.no_grad() to remove gradient computations
    # This is not necessary, but can be much faster
    with torch.no_grad():
        for features,labels in dataloader:
            # Ensure features and labels are on the correct device
            features = features.to(device)
            labels = labels.to(device)
            # Make predictions
            out = net(features)
            # Compute loss and accuracy of the model
            loss += loss_fn(out,labels)
            pred = torch.max(out,1)[1]
            acc += (pred==labels).sum()
            count += len(labels)
    return loss.item()/count, acc.item()/count

def train(net, train_loader, test_loader, device=torch.device('cpu'), optimiser=None, lr=0.01, epochs=10, loss_fn=nn.NLLLoss()):
    optimiser = optimiser or torch.optim.Adam(net.parameters(),lr=lr)
    res = { 'train_loss' : [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(epochs):
        tl,ta = train_epoch(
            net=net,
            dataloader=train_loader,
            device=device,
            optimiser=optimiser, lr=lr, loss_fn=loss_fn)
        vl,va = validate_epoch(
            net=net,
            dataloader=test_loader,
            device=device,
            loss_fn=loss_fn)
        print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res

def main():

    net = MyNet()
    currrent_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {currrent_device}")
    summary(net, input_size=(1,28,28),device=currrent_device)
    hist = train(net, train_loader, test_loader, device=torch.device(currrent_device), epochs=5)
    plot_results(hist)
    plt.show()

if __name__ == "__main__":
    main()