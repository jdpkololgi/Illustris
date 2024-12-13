import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from Network_stats import network


DEVICE = torch.device('mps') # Apple Silicon available
BATCHSIZE = 16
CLASSES = 4
EPOCHS = 50 #no need to run a full experiment
LOG_INTERVAL = 10 # print training status every 10 epochs
N_TRAIN_EXAMPLES = BATCHSIZE * 30 # no need to use the full dataset - risk of overfitting
N_VALID_EXAMPLES = BATCHSIZE * 10

# Load dataset
def load_CW_data():
    _net = network()
    _net.pipeline(network_type='Delaunay') # {MST, Complex, Delaunay}
    return _net.train_loader, _net.val_loader, _net.class_weights


# Define MLP model
def define_model(trial):
    # Optimise number of layers, neurons and dropout ratio in each layer
    n_layers = trial.suggest_int('n_layers', 1, 5) # number of layers will be between 1 and 5
    layers = [] # list to store layers

    in_features = 7 # input features are 7 for the Delaunay network
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128, log = True) # number of neurons in each layer will be between 4 and 128
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)  # dropout ratio will be between 0.2 and 0.5
        layers.append(nn.Dropout(p))

        in_features = out_features
    
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def objective(trial, train_loader, val_loader, class_weights):
    # generate model
    model = define_model(trial).to(DEVICE) # create instance of model and move to apple silicon gpu
    
    # generate optimiser
    lr = trial.suggest_float('lr', 1e-7, 1e-1, log=True) # learning rate will be between 1e-7 and 1e-1
    optimiser_name = trial.suggest_categorical('optimiser', ['Adam', 'RMSprop', 'SGD']) # optimiser will be one of Adam, RMSprop or SGD
    optimiser_class = getattr(optim, optimiser_name) # get optimiser class object
    optimiser = optimiser_class(model.parameters(), lr=lr) # create instance of optimiser

    # choose weighted loss function
    loss_fn = trial.suggest_categorical('loss_fn', [F.cross_entropy, F.nll_loss]) # loss function will be one of CrossEntropyLoss or NLLLoss

    # train model
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # limit training set size
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE) # move data to apple silicon gpu

            optimiser.zero_grad()
            output = model(data)
            loss = loss_fn(output, target, weight=class_weights)
            loss.backward()
            optimiser.step()

        # validate model
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # limit validation set size
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break

                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(val_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy

if __name__ == '__main__':

    train_loader, val_loader, class_weights = load_CW_data()
    study = optuna.create_study(direction='maximize') # as objective function outputs accuracy, we want to maximise this
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, class_weights), n_trials=100, timeout=600) # run 100 trials or until 10 minutes have passed
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print(study.trials_dataframe())
    study.trials_dataframe().to_csv('optuna_results.csv') # save results to csv file