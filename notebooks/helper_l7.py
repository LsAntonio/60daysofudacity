from sklearn.preprocessing import StandardScaler
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def generate_data(shuffle = True, normalize = True):
    """
    Generate data from the iris data set.
    
    params:
    ------
    shuffle -- Shuffle or not the data.
    
    returns:
    -------
    X, y    -- Data and labels (tensors).
    """
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    if shuffle:
        # random shuffle
        index = np.arange(X.shape[0]) # samples
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    # convert data to tensor
    X = T.from_numpy(X).float()
    y = T.tensor(y.astype(float), requires_grad = True).long()
    return X, y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.output(x), dim = 1)
        return x

def update_model(server_model, m1, m2, m3, m4):
    """
    Average the parameters (weight and bias) of the model on a external worker.
    
    params:
    ------
    server_model -- Model located on the main server.
    m1, ..., m4  -- Models (pointers).
    
    returns:
    -------
    server_model -- Updated server model.
    
    """
    with T.no_grad():
        server_model.fc1.weight.set_(((m1.fc1.weight.data + m2.fc1.weight.data + m3.fc1.weight.data + m4.fc1.weight.data) / 4).get())
        server_model.fc1.bias.set_(((m1.fc1.bias.data + m2.fc1.bias.data + m3.fc1.bias.data + m4.fc1.bias.data) / 4).get())
        
        server_model.fc2.weight.set_(((m1.fc2.weight.data + m2.fc2.weight.data + m3.fc2.weight.data + m4.fc2.weight.data) / 4).get())
        server_model.fc2.bias.set_(((m1.fc2.bias.data + m2.fc2.bias.data + m3.fc2.bias.data + m4.fc2.bias.data) / 4).get())
        
        server_model.output.weight.set_(((m1.output.weight.data + m2.output.weight.data + m3.output.weight.data + m4.output.weight.data) / 4).get())
        server_model.output.bias.set_(((m1.output.bias.data + m2.output.bias.data + m3.output.bias.data + m4.output.bias.data) / 4).get())
    print('Main model updated!')
    return server_model

def train_worker(model, X, y, opt, criterion):
    """
    Train a model on a worker.
    
    params:
    ------
    model     -- Model pointer.
    X, y      -- Data and label pointers.
    opt       -- optimiezer.
    criterion -- Model criterion.
    
    returns:
    -------
    loss      -- Model loss (pointer).
    """
    # set zero grads
    opt.zero_grad()
    # y_hat
    y_hat = model(X)
    # calculate loss
    loss = criterion(y_hat, y)
    # backward
    loss.backward()
    # update
    opt.step()
    return loss

def federated_avg_training(epochs, lr, worker_1, worker_2, worker_3, worker_4, worker_5, x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Train a model over four workers using Federated Learning.
    
    params:
    ------
    epochs       -- Training epochs.
    
    returns:
    -------
    server_model -- Trained (average) model.
    """
    loss_training = []
    worker_1_loss = []
    worker_2_loss = []
    worker_3_loss = []
    worker_4_loss = []
    iters = 1
    server_model = Net()
    for epoch in range(epochs):
        model_w1 = server_model.copy().send(worker_1)
        model_w2 = server_model.copy().send(worker_2)
        model_w3 = server_model.copy().send(worker_3)
        model_w4 = server_model.copy().send(worker_4)
        opt_w1 = optim.SGD(model_w1.parameters(), lr = lr)
        opt_w2 = optim.SGD(model_w2.parameters(), lr = lr)
        opt_w3 = optim.SGD(model_w3.parameters(), lr = lr)
        opt_w4 = optim.SGD(model_w4.parameters(), lr = lr)
        criterion_w1 = nn.NLLLoss().send(worker_1)
        criterion_w2 = nn.NLLLoss().send(worker_2)
        criterion_w3 = nn.NLLLoss().send(worker_3)
        criterion_w4 = nn.NLLLoss().send(worker_4)
        epoch_loss = 0.0
        for inner_iter in range(iters):
            loss_w1 = train_worker(model_w1, x1, y1, opt_w1, criterion_w1).get().detach().numpy()
            loss_w2 = train_worker(model_w2, x2, y2, opt_w2, criterion_w2).get().detach().numpy()
            loss_w3 = train_worker(model_w3, x3, y3, opt_w3, criterion_w3).get().detach().numpy()
            loss_w4 = train_worker(model_w4, x4, y4, opt_w4, criterion_w4).get().detach().numpy()
            epoch_loss = (loss_w1 + loss_w2 + loss_w3 + loss_w4)/4
            worker_1_loss.append(loss_w1)
            worker_2_loss.append(loss_w2)
            worker_3_loss.append(loss_w3)
            worker_4_loss.append(loss_w4)
            loss_training.append(epoch_loss)
        model_w1.move(worker_5)
        model_w2.move(worker_5)
        model_w3.move(worker_5)
        model_w4.move(worker_5)
        # sum-up gradients on worker 5 and retrieve the model
        server_model = update_model(server_model, model_w1, model_w2, model_w3, model_w4)
        print('Epoch: {}|{} Avg loss: {}'.format(epoch, epochs, epoch_loss))
    return server_model, loss_training, worker_1_loss, worker_2_loss, worker_3_loss, worker_4_loss

def plot_losses(epochs, losses, name, average = False):
    """
    Plot the losses.
    """
    f, arr = plt.subplots(1,2, figsize = (15, 5))
    arr[0].plot(range(0, epochs), losses[0])
    if average:
        arr[0].set_title('Average (server) Training loss')
        name = name + '_avg'
    else:
        arr[0].set_title('Server Training loss')
    arr[0].set_xlabel('epochs')
    arr[0].set_ylabel('loss')
    arr[1].plot(range(0, epochs), losses[1])
    arr[1].plot(range(0, epochs), losses[2])
    arr[1].plot(range(0, epochs), losses[3])
    arr[1].plot(range(0, epochs), losses[4])
    arr[1].set_title('Training losses per worker')
    arr[1].legend(['W1 loss', 'W2 loss', 'W3 loss', 'W4 loss'])
    arr[1].set_xlabel('epochs')
    arr[1].set_ylabel('loss')
    f.savefig('./' + name + '_fl.png', format = 'png', dpi = 250, bbox_inches = 'tight')
    
def eval_model(model, X, y):
    model.eval()
    y_hat = model(X)
    y_hat.argmax(dim = 1)
    print('Training accuracy: {}'.format(accuracy_score(y.numpy().astype(int), y_hat.argmax(dim = 1).detach().numpy().astype(int))))
