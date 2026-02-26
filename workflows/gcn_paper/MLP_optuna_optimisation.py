import sys
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print("usage: MLP_optuna_optimisation.py [--help]\n\nRuns the legacy Optuna-based MLP tuning workflow.")
    raise SystemExit(0)

import datetime
import logging

try:
    import optuna
    from optuna.trial import TrialState
    import optuna.visualization as ov
except ImportError:
    optuna = None
    TrialState = None
    ov = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from Network_stats import network


DEVICE = torch.device('mps') # Apple Silicon available
# BATCHSIZE = 16
CLASSES = 4
EPOCHS = 100 #no need to run a full experiment
LOG_INTERVAL = 10 # print training status every 10 epochs
BATCHSIZE = 16 # batch size will be between 8 and 64
N_TRAIN_EXAMPLES = BATCHSIZE * 400 # no need to use the full dataset - risk of overfitting
N_VALID_EXAMPLES = BATCHSIZE * 200
LR = 1e-5

# Load dataset
def load_CW_data():
    _net = network()
    _net.pipeline(network_type='Delaunay') # {MST, Complex, Delaunay} # _net.pipeline_from_save(network_type='Delaunay') # {MST, Complex, Delaunay}
    return _net.train_loader, _net.val_loader, torch.tensor(_net.class_weights, dtype=torch.float32).to(DEVICE) # move class weights to apple silicon gpu

# Define MLP model
def define_model(trial):
    # Optimise number of layers, neurons and dropout ratio in each layer
    n_layers = trial.suggest_int('n_layers', 1, 5) # number of layers will be between 1 and 5
    layers = [] # list to store layers

    # activation_fn_name = trial.suggest_categorical('activation_fn', ['ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh']) # activation function will be one of ReLU, LeakyReLU, Linear, Sigmoid or Tanh
    # activation_fn = getattr(nn, activation_fn_name) # get activation function class object
    in_features = 7 # input features are 7 for the Delaunay network
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 15, log = True) # number of neurons in each layer will be between 4 and 25 (was 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)  # dropout ratio will be between 0.2 and 0.5
        # layers.append(nn.Dropout(p))

        in_features = out_features
    
    layers.append(nn.Linear(in_features, CLASSES))
    # layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def objective(trial, train_loader, val_loader, class_weights):
    # generate model
    model = define_model(trial).to(DEVICE) # create instance of model and move to apple silicon gpu
    
    # generate optimiser
    lr = LR #trial.suggest_float('lr', 1e-7, 1e-4, log=True) # learning rate will be between 1e-7 and 1e-1
    optimiser_name = trial.suggest_categorical('optimiser', ['Adam', 'RMSprop', 'SGD']) # optimiser will be one of Adam, RMSprop or SGD
    optimiser_class = getattr(optim, optimiser_name) # get optimiser class object
    optimiser = optimiser_class(model.parameters(), lr=lr) # create instance of optimiser
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) # use AdamW optimiser with weight decay

    # choose weighted loss function
    loss_fn_name = 'cross_entropy'#trial.suggest_categorical('loss_fn', ['cross_entropy', 'nll_loss']) # loss function will be one of CrossEntropyLoss or NLLLoss
    loss_fn = getattr(F, loss_fn_name) # get loss function class object
    
    # choose batch size
    # BATCHSIZE = trial.suggest_int('batch_size', 8, 64, log=True) # batch size will be between 8 and 64
    # N_TRAIN_EXAMPLES = BATCHSIZE * 30 # no need to use the full dataset - risk of overfitting
    # N_VALID_EXAMPLES = BATCHSIZE * 10
    
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

    return float(accuracy)

if __name__ == '__main__':
    if optuna is None:
        raise ImportError("optuna is required to run this script. Install with `pip install optuna`.")
    # sampler = optuna.samplers.
    train_loader, val_loader, class_weights = load_CW_data()
    study = optuna.create_study(direction='maximize', storage='sqlite:///optuna_study.db', load_if_exists=True, study_name=f'MLP_BS_{BATCHSIZE}_{LR}') # as objective function outputs accuracy, we want to maximise this
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, class_weights), n_trials=500, timeout=600) # run 100 trials or until 10 minutes have passed
    
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
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    study.trials_dataframe().to_csv(f'optuna_results_{current_time}.csv') # save results to csv file

    # visualise results
    ov.plot_optimization_history(study).show() # plot optimisation history
    ov.plot_intermediate_values(study).show() # plot intermediate values
    ov.plot_parallel_coordinate(study).show() # visualising higher dimensional parameter spaces
    ov.plot_parallel_coordinate(study, params=['optimiser', 'loss_fn']).show() # visualising higher dimensional parameter spaces
    ov.plot_param_importances(study).show() # plot parameter importances
    ov.plot_contour(study, params=['n_units_l0', 'n_units_l1']).show() # plot contour plot
    ov.plot_slice(study).show() # plot slice plot
    ov.plot_slice(study, params=['n_units_l0', 'n_units_l1']).show() # plot slice plot
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler()) # add stream handler to logger