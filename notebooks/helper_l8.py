## Libraries ##
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import torch as T
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
###############
T.manual_seed(5)
np.random.seed(7)

# Helper Functions
def generate_data(shuffle = True):
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
    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if shuffle:
        # random shuffle
        index = np.arange(X.shape[0]) # samples
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
    # convert data to tensor
    X = T.from_numpy(X).float()
    y = T.tensor(y.astype(float), requires_grad = True).long()
    return X, y

class Net(nn.Module):
    """
    A shallow neural network.
    """
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

def train_worker(model, X, y, opt):
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
    model     -- Trained model.
    loss      -- Model loss (pointer).
    """
    criterion = nn.NLLLoss().send(X.location)
    model.send(X.location)
    opt.zero_grad()
    y_hat = model(X)
    loss = criterion(y_hat, y)
    loss.backward()
    opt.step()
    return model, loss

def plot_data(X, y, feature = 0):
    """
    Explore the relationship of the first feature aga the data set.
    
    params:
    ------
    X       -- Tensor data.
    y       -- Tensor labels.
    feature -- Feature to exaplore (0, ..., 3)
    """
    n = [0, 1, 2, 3]
    n.remove(feature)
    features_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    f, arr = plt.subplots(1, 3, figsize = (15, 5))
    f.suptitle('Relationship of feture {} agains the others'.format(features_names[feature]))
    for x in range(3):
        arr[x].scatter(X[:, feature].numpy(), X[:, n[x]].numpy(), c = y[:], edgecolors = 'b')
        arr[x].set_xlabel(features_names[feature])
        arr[x].set_ylabel(features_names[x])

def create_remote_dataset(workers, data_loader):
    """
    Creates a remote data set.
    
    params:
    ------
    workers      -- List of workers.
    data_loader  -- Data loader.
    """
    remote_dataset = (list(),list())
    for batch_idx, (data,target) in enumerate(data_loader):
        data = data.send(workers[batch_idx % len(workers)])
        target = target.send(workers[batch_idx % len(workers)])
        remote_dataset[batch_idx % len(workers)].append((data, target))
    return remote_dataset

def average_workers(loss):
    """
    Average losses per worker.
    
    params:
    ------
    loss   -- Array with losses.
    
    returns:
    -------
    Averaged losses.
    """
    avg = 0
    for l in loss:
        avg += l
    return avg/len(loss)

def update_server_model(server_model, params):
    """
    Average the parameters (weight and bias) of the model on a external worker.
    
    params:
    ------
    server_model -- Model located on the main server.
    params       -- Average params.
    
    returns:
    -------
    server_model -- Updated server model.
    
    """
    with T.no_grad():
        server_model.fc1.weight.set_(params[0])
        server_model.fc1.bias.set_(params[1])
    
        server_model.fc2.weight.set_(params[2])
        server_model.fc2.bias.set_(params[3])
    
        server_model.output.weight.set_(params[4])
        server_model.output.bias.set_(params[5])
    print('Main model updated!')
    return server_model

def federated_training(epochs, remote_dataset, models, optimizers, workers, secure_worker):
    """
    This function train a federated model using trusted aggregation.
    
    params:
    ------
    epochs    -- Total epoch to train.
    
    """
    server_model = Net()
    w1_loss = list()
    w2_loss = list()
    avg_loss = list()
    params = [list(models[0].parameters()), list(models[1].parameters())]
    for epoch in range(epochs):
        losses = {}
        losses[0] = []
        losses[1] = []
        for data_index in range(len(remote_dataset[0])-1):
            for remote_index in range(len(workers)):
                X, y = remote_dataset[remote_index][data_index]
                models[remote_index], loss = train_worker(models[remote_index], X, y, optimizers[remote_index])
                losses[remote_index].append(loss.get().detach().numpy())
            new_params = list()
            for param_i in range(len(params[0])):
                spdz_params = list()
                for remote_index in range(len(workers)):
                    spdz_params.append(params[remote_index][param_i].copy().fix_precision().share(workers[0], workers[1], crypto_provider=secure_worker).get())

                new_param = (spdz_params[0] + spdz_params[1]).get().float_precision()/2
                new_params.append(new_param)
            with T.no_grad():
                for model in params:
                    for param in model:
                        param *= 0

                for model in models:
                    model.get()

                for remote_index in range(len(workers)):
                    for param_index in range(len(params[remote_index])):
                        params[remote_index][param_index].set_(new_params[param_index])
        w1_loss.append(average_workers(losses[0]))
        w2_loss.append(average_workers(losses[1]))
        avg_loss.append((w1_loss[epoch] + w2_loss[epoch])/2)
        print('Epoch: {}|{} | Avg Loss: {}'.format(epoch + 1, epochs, avg_loss[epoch]))
    server_model = update_server_model(server_model, new_params)
    return server_model, w1_loss, w2_loss, avg_loss

def plot_losses(w1_loss, w2_loss, avg_loss):
    f, arr = plt.subplots(2, 2, figsize = (10, 7))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.suptitle('Evaluation loss')
    epochs = range(len(w1_loss))
    losses = [w1_loss, w2_loss, avg_loss]
    info = ['Worker 1 loss', 'Worker 2 loss', 'Server loss', 'Models losses']
    colors = ['g', 'b', 'r']
    for r in range(2):
        for c in range(2):
            if r and c == 1:
                for fig in range(3):
                    arr[r, c].plot(epochs, losses[fig])
                arr[r, c].legend([info[0], info[1], info[2]])
                arr[r, c].set_xlabel('epochs')
                arr[r, c].set_ylabel('loss')
                arr[r, c].set_title(info[(c + r) + r])
                
            else:
                arr[r, c].plot(epochs, losses[(c + r) + r], c = colors[(c + r) + r])
                if r == 1 and c == 0:
                    arr[r, c].set_xlabel('epochs')
                arr[r, c].set_ylabel('loss')
                arr[r, c].set_title(info[(c + r) + r])    
    
    
def evaluate(X, y, model):
    """
    Evaluate the model.
    
    params:
    ------
    X      -- Train data.
    y      -- Labels.
    model  -- Model.
    
    returns:
    -------
    Accuracy.
    """
    with T.no_grad():
        y_hat = model(X)
    print('Training accuracy: {}'.format(accuracy_score(y, y_hat.argmax(dim = 1).numpy())))
