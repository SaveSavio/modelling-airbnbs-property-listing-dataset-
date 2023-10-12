from typing import Any
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tabular_data import load_airbnb
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
import json
import os
from datetime import datetime
from sklearn.metrics import r2_score
import itertools
import typing
import glob
import numpy as np
import time


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__() # this bit is needed when inheriting so to initialize the parent class methods
        df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
        self.features, self.labels = load_airbnb(df, label="Price_Night", numeric_only=True)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx]).float() # simply gets the idx example and transfor it into a torch object
        label = torch.tensor(self.labels.iloc[idx]).float()
        return (features, label)
    
    def __len__(self):
        return len(self.features)


def data_loader(dataset, train_ratio=0.7, val_ratio=0.15, batch_size=32, shuffle=True):    
    # TODO: confirm batch is necessary, cause I am splitting also train and test sets???
    dataset_size = len(dataset)
    # Calculate the number of samples for each split
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Use random_split to split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # not so nice to see...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=shuffle) # use full batch for validation and testing
    #TODO: change val to validation
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader


class NN(torch.nn.Module):
    """
        Define a fully connected neural network.
        On initialization, set the following
            Parameters:
                input_dim: the dimension of the input layer
                output dim:
                dept: Depth of the model (number of hidden layers)
                width: Width of each hidden layer (all hidden layers have the same width)
    """
    def __init__(self, input_dim=9, output_dim=1, depth=1, width=9, **kwargs):
        super().__init__()
        # define layers
        self.layers = torch.nn.Sequential()
        
        # Input layer
        self.layers.add_module("input_layer", torch.nn.Linear(input_dim, width))
        self.layers.add_module("relu1",  torch.nn.ReLU())

        # Hidden layers
        for i in range(depth - 1):
            self.layers.add_module(f"hidden_layer{i}", torch.nn.Linear(width, width))
            self.layers.add_module(f"relu{i + 1}",  torch.nn.ReLU())

        # Output layer
        self.layers.add_module("output_layer",  torch.nn.Linear(width, output_dim))


    def forward(self, X):
        # return prediction
        return self.layers(X)


def train(model, epochs = 10, optimizer='Adam', **kwargs): ## TODO: consider if we have to pass train_loader to it
    if optimizer == 'Adam': # TODO: should implement alternative optimizers
        # torch provides model parameters throught the .parameters() method
        # this method is inherited from the torch class
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    writer = SummaryWriter() # Tensorboard class init

    batch_idx = 0 # initalize outside the epoch
    training_time = 0
    cumulative_inference_latency = 0
    
    for epoch in range(epochs):
        print("\nepoch: ", epoch, "/", epochs)
        training_start_time = time.time()

        for batch in train_loader:
            features, labels = batch
            prediction = model(features) # forward step
            loss = F.mse_loss(prediction, labels)
            loss.backward() # differentiate the loss
            #print("mse: ", loss.item())
            R_squared = r_squared(prediction, labels)
            # TODO: The time taken to train the model under a key called training_duration
            # TODO: The average time taken to make a prediction under a key called inference_latency
            optimizer.step() # optimization step
            optimizer.zero_grad() # set gradient values to zero before another backwards step
            writer.add_scalar('loss', loss.item(), batch_idx) # TensorFlow
            batch_idx += 1 # using a separate index so that it does not reset every epoch as "batch" does

        training_stop_time = time.time()
        training_time += training_stop_time - training_start_time
        print("training time: ", training_time)

        for batch in val_loader:
            
            features, labels = batch
            inference_start_time = time.time()
            prediction = model(features)
            inference_stop_time = time.time()
            cumulative_inference_latency += inference_stop_time - inference_start_time
            validation_loss = F.mse_loss(prediction, labels) # check the validation loss every epoch
            print("validation mse: ", validation_loss.item())
            writer.add_scalar('validation loss', validation_loss.item(), epoch) # TensorFlow
    
    average_inference_latency = cumulative_inference_latency/epochs
    print("average inference latency:", average_inference_latency)
    return loss.item(), R_squared, training_time, average_inference_latency


def get_nn_config(config_file_path='nn_config.yaml'):
    """
        Reads the neural network configuration from a YAML file and returns it as a dictionary.
        
        Parameters:
            config_file_path (str): Path to the YAML configuration file.
        Returns:
            dict: A dictionary containing the configuration settings.
    """
    try:
        with open(config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the configuration file: {str(e)}")


def save_model(model, config, RMSE_loss = None, R_squared = None, training_duration = None, inference_latency = None):
    
    current_time = datetime.now()    # Get the current date and time
    folder_name = current_time.strftime('%Y-%m-%d_%H.%M.%S') # Format the current time as a string in the desired format

    model_path = './neural_networks/regression/' + folder_name
    os.mkdir(model_path)     # Create a directory with the formatted name
    torch.save(model.state_dict(), model_path+'/model.pt')
    
    hyperparameters_file = '/hyperparameters.json'
    metrics_file = '/metrics.json'

    metrics = {'RMSE_loss': RMSE_loss, 'R_squared': R_squared, 'training_duration': training_duration, 'interference_latency': inference_latency}

    with open(model_path+hyperparameters_file, 'w') as json_file:
        json.dump(config, json_file) 

    with open(model_path+metrics_file, 'w') as json_file:
        json.dump(metrics, json_file) 


def r_squared(predictions, labels):
    """
    Calculate the R-squared (coefficient of determination) between predictions and labels.

    Args:
        predictions (torch.Tensor): Tensor containing predicted values.
        labels (torch.Tensor): Tensor containing true labels.

    Returns:
        float: R-squared score.
    """
    mean_labels = torch.mean(labels)
    total_sum_of_squares = torch.sum((labels - mean_labels)**2)
    residual_sum_of_squares = torch.sum((labels - predictions)**2)
    
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    
    return r2.item()


def generate_nn_configs(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


def find_best_nn(grid):

    hyperparameters_grid = generate_nn_configs(grid) # generate the grid

    for config in hyperparameters_grid: # loop through th grid
        print(config)
        loss, R_squared, training_time, inference_latency = train(model, **config) # determine the loss for each hyperparam configuration
        save_model(model, config, RMSE_loss = loss, R_squared=R_squared, training_duration=training_time, inference_latency=inference_latency) # save the model
    
        #TODO: now find the best model
        #TODO: save the best model in a folder


#TODO: Complete the training loop so that it iterates through every batch in the dataset for the specified number of epochs,
# and optimises the model parameters.
# You should add a step to evaluate the model performance on the validation dataset after each epoch.

#TODO: Use tensorboard to visualize the training curves of the model and the accuracy both on the training and validation set.

if __name__ == "__main__":
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_loader, val_loader, test_loader = data_loader(dataset, batch_size=32, shuffle=True)

    batch_size = len(dataset)  # Full batch
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # TODO: what was this here for?

    config = get_nn_config()
    model = NN(**config)
    
    grid = {
    "learning_rate": [0.01],
    "depth": [2, 3],
    "hidden layer width": [8, 16],
    "batch size": [32],
    "epochs": [10]
    }

    find_best_nn(grid=grid)

    #loss, R_squared = train(model, **config) # this is here in case I do not want to use get_nn_config

    #save_model(model, config, RMSE_loss = loss, R_squared=R_squared)   