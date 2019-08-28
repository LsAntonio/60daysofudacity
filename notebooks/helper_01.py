# Imports
from sklearn import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
np.random.seed(5)
torch.manual_seed(5)

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score

####################
# Helper Functions #
####################

cls_names = ['AdaBoostClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'MLPClassifier',
             'DecisionTreeClassifier', 'SVC', 'GradientBoostingClassifier', 'SGDClassifier', 'LogisticRegression',
             'RidgeClassifier']

def plot_digits(X, y):
    """
    Plot the digit data.
    
    params:
    ------
    X       -- Data set (tensor).
    y       -- Labels (tensor).
    """
    f, arr = plt.subplots(2, 5, figsize = (15, 5))
    for r in range(2):
        for c in range(5):
            sample = int(np.random.randint(0, X.size(0), 1))
            arr[r,c].imshow(X[sample].reshape(8, 8).numpy(), aspect = 'auto')
            arr[r,c].set_title('Number: {}'.format(y[sample].numpy()))
            arr[r,c].axis('off')
    f.savefig('./plots/digits.png', format = 'png', dpi = 250, bbox_inches = 'tight')

def generate_data(shuffle = True, noise = False, mu = 0, sigma = 0.1):
    """
    Generate data from the iris data set.
    
    params:
    ------
    shuffle -- Shuffle or not the data.
    
    returns:
    -------
    X, y    -- Data and labels (tensors).
    """
    X, y = datasets.load_digits(return_X_y = True)
    if noise:
        data_noise = np.random.normal(mu, sigma, X.shape)
        X = X + data_noise
    if shuffle:
        # random shuffle
        index = np.arange(X.shape[0]) # samples
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
    # create 10-rounds
    count = X.shape[0]//11
    # 
    remote_datasets = dict()
    for k in range(10):
        remote_datasets[k] = {'data': X[k * count: (k * count) + count], 'labels': y[k * count: (k * count) + count]}
    # convert data to tensor for the pytorch model.
    X = torch.from_numpy(X[10 * count:]).float()
    y_true = torch.tensor(y[10 * count:].astype(float), requires_grad = True).long()
    return remote_datasets, X, y_true

def train_classifiers(datasets, print_results = True):
    """
    Trains 10 classifiers on each dataset.
    
    params:
    ------
    dataset     -- Dataset dictionary.
    
    returns:
    -------
    classifiers -- Dictionary, containing the training classifiers.
    """
    classifiers = dict()
    # set random generator for each classifier
    seeds = np.random.randint(0, 20, 10)
    #
    classifiers[0] = AdaBoostClassifier(random_state = seeds[0])
    classifiers[1] = RandomForestClassifier(n_estimators = 100, random_state = seeds[1])
    classifiers[2] = KNeighborsClassifier() #random_state = seeds[2]
    classifiers[3] = MLPClassifier(random_state = seeds[3], max_iter = 500)
    classifiers[4] = DecisionTreeClassifier(random_state = seeds[4])
    classifiers[5] = SVC(gamma = 'scale', random_state = seeds[5], probability = True)
    classifiers[6] = GradientBoostingClassifier(random_state = seeds[6])
    classifiers[7] = SGDClassifier(max_iter = 1000, tol = 1e-3, random_state = seeds[7])
    classifiers[8] = LogisticRegression(multi_class = 'auto', solver = 'lbfgs', random_state = seeds[8], max_iter = 700)
    classifiers[9] = RidgeClassifier(random_state = seeds[9])
    # Training
    for k in range(len(classifiers)):
        classifiers[k].fit(datasets[k]['data'], datasets[k]['labels'])
    # Report accuracy
    if print_results:
        for k in range(len(classifiers)):
            acc = accuracy_score(datasets[k]['labels'], classifiers[k].predict(datasets[k]['data']))
            print('[{}] - {} classifier | accuracy: {}'.format(k + 1, cls_names[k], acc))
    return classifiers

def generate_predictions(X, classifiers):
    """
    Generates a set of predictions from the classifiers.
    
    params:
    ------
    dataset     -- Dictionary, containing the generated datasets.
    classifiers -- Dictionary, containing the trained classifiers.
    
    returns:
    -------
    preds       -- Predictions.
    """
    preds = np.empty((X.shape[0], len(classifiers)))
    for k in range(len(classifiers)):
        preds[:, k] = classifiers[k].predict(X.numpy())
    return preds

def generate_labels(preds, epsilon = 0.1):
    """
    Generates labels for the current model from the predictions adding
    laplace noise.
    """
    new_labels = list()
    for pred in preds:
        label_count = np.bincount(pred.astype(int), minlength = 10)
        beta = 1/epsilon
        for i in range(len(label_count)):
            label_count[i] += np.random.laplace(0, beta, 1)
        new_label = np.argmax(label_count)
        new_labels.append(new_label)
    return new_labels

def calculate_accuracy(model, X, y, show_acc = True):
    """
    Calculate the accuracy of a model.
    
    params:
    ------
    model   -- Trained model.
    X       -- Dataset.
    y       -- Labels.
    show_acc-- Print the accuracy.
    
    returns:
    -------
    accuracy
    """
    y_pred = model(X)
    # converting probabilities into target
    ps = torch.exp(y_pred)
    top_p, top_class = ps.topk(1, dim=1)
    acc = accuracy_score(top_class.squeeze(1).numpy(), y.numpy())
    if show_acc:
        print('Accuracy: {}'.format(acc))
    return acc

def run_experiment(epsilons, X, y_true, preds, epochs = 50, print_results = True):
    criterion = nn.NLLLoss()
    y_single = list()
    p_single = list()
    for epsilon in epsilons:
        # generate new labels
        new_labels = generate_labels(preds, epsilon = epsilon)
        # convert labels
        labels = np.array(new_labels)
        ylabels = torch.tensor(labels.astype(float), requires_grad = True).long()
        # train the model
        model = Net()
        # optimizer
        op = optim.Adam(model.parameters(), lr = 0.01)
        model = train_model(model, X, ylabels, op, criterion, epochs, print_epoch = False)
        y_single.append(calculate_accuracy(model, X, y_true, False))
        p_single.append(calculate_accuracy(model, X, ylabels, False))
        if print_results:
            print('Epsilon: {} | Accuracy on real labels: {} | Accuracy on generated labels: {}'.format(epsilon,
                                                                                                        calculate_accuracy(model, X, y_true, False),
                                                                                                        calculate_accuracy(model, X, ylabels, False)))
    return y_single, p_single

def run_experiment_avg(epsilons, X, y_true, preds, average = 20, epochs = 50, print_results = True):
    criterion = nn.NLLLoss()
    y_avg = list()
    p_avg = list()
    for epsilon in epsilons:
        avg_true_y = 0.0
        avg_pred_y = 0.0
        for avg in range(average)
            # generate new labels
            new_labels = generate_labels(preds, epsilon = epsilon)
            # convert labels
            labels = np.array(new_labels)
            ylabels = torch.tensor(labels.astype(float), requires_grad = True).long()
            # train the model
            model = Net()
            # optimizer
            op = optim.Adam(model.parameters(), lr = 0.01)
            model = train_model(model, X, ylabels, op, criterion, 50, print_epoch = False)
            # accumulate
            avg_true_y += calculate_accuracy(model, X, y_true, False)
            avg_pred_y += calculate_accuracy(model, X, ylabels, False)
        avg_true_y = avg_true_y/average
        avg_pred_y = avg_pred_y/average
        y_avg.append(avg_true_y)
        p_avg.append(avg_pred_y)
        if print_results:
            print('Epsilon: {} | Avg accuracy on real labels: {} | Avg Accuracy on generated labels: {}'.format(epsilon, avg_true_y, avg_pred_y))
    return y_avg, p_avg

def comparison_plot(epsilons, single, average, name):
    """
    Plots a comparison using a single run vs an average
    
    params:
    ------
    epsilons    -- Range of epsilons.
    single      -- List of single run.
    average     -- List of average runnings.
    
    returns:
    -------
    Plot.
    """
    f, arr = plt.subplots(1, 2, figsize = (15, 5))
    arr[0].plot(np.log(epsilons), single[0])
    arr[0].plot(np.log(epsilons), single[1])
    arr[0].legend([r'$\tilde{y} $ true', r'$ \hat{y} $ pred'])
    arr[0].set_xlabel(r'$\log{\epsilon}}$')
    arr[0].set_ylabel('Accuracy')
    arr[0].set_title('Single Run Accuracy')
    arr[1].plot(np.log(epsilons), average[0])
    arr[1].plot(np.log(epsilons), average[1])
    arr[1].legend([r'$\tilde{y} $ true', r'$ \hat{y} $ pred'])
    arr[1].set_xlabel(r'$\log{\epsilon}}$')
    arr[1].set_ylabel('Accuracy')
    arr[1].set_title('Average Run Accuracy')
    f.savefig('../plots/' + name + '.png', format = 'png', bbox_inches = 'tight', dpi = 250)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 50)
        self.fc2 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.output(x), dim = 1)
        return x

def train_model(model, X, y, model_optimizer, criterion, n_epochs, print_epoch = True):
    model.train()
    model.to('cpu')
    for epoch in range(n_epochs):
        model_optimizer.zero_grad()
        # get predictions
        pred = model(X)
        # calculate loss
        loss = criterion(pred, y)
        # backpropagation
        loss.backward()
        # optimize
        model_optimizer.step()
        # show loss
        if print_epoch:
            print('Epoch: {} | {}, loss: {}'.format(epoch + 1, n_epochs, loss.data.numpy()))
    return model
