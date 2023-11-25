from datetime import datetime
import glob
import itertools
import json
import numpy as np
import os
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.functional import r2_score
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tabular_data import database_utils as dbu
import typing
from typing import Any
import yaml

# TODO: consider if this can be commonized between Classification & Regression
class AirbnbNightlyPriceClassificationDataset(Dataset): 
    """
        Creates a PyTorch dataset from tabular data.
        Parameters:
            On initialization, requires
            - the dataset path (csv file)
            - the index of the label column
        Returns:
            two Torch tensor in float precision 
            - a Tensor with the numerical tabular features of the house
            - an array of features
    """
    def __init__(self, dataset_path, label):
        super().__init__() # initializes Dataset methods
        df = pd.read_csv(dataset_path)
        self.features, self.labels = dbu.load_airbnb(df, label=label, numeric_only=True)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx]).float() # simply gets the idx example and transfor it into a torch object
        label = torch.tensor(self.labels.iloc[idx])
        return (features, label)
    
    def __len__(self):
        return len(self.features)


def data_loader(dataset, train_ratio=0.7, validation_ratio=0.15, test_batch_size=32, shuffle=True):    
    """
        Dataloader function that
            - splits the data into test, train and validation datasets
            - shuffles the data
            - generates batches of data
        Parameters:
            - dataset (an instance of Pytorch DataSet class)
            - train and validation ratios
            - batch size (for use in the DataLoader)
            - shuffle (if data shuffling is required)

    It uses full batch for validation and testing.
    """

    # Calculate the number of samples for each split
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    validation_size = int(validation_ratio * dataset_size)
    test_size = dataset_size - train_size - validation_size

    # Use random_split to split the dataset
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # use torch DataLoader on all sets
    train_loader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=shuffle)
    # use full batch for validation and testing
    validation_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, validation_loader, test_loader


class Classification(torch.nn.Module):
    """
        Defines a fully connected neural network. Inherits methods from the class "torch.nn.Module".
        On initialization, set the following
            Parameters:
                input_dim: the dimension of the input layer (number of features)
                output dim: the number of labels to be predicted
                dept: Depth of the model, i.e. the number of hidden layers
                width: Width of each hidden layer (all hidden layers have the same width)
            Returns:
                when called on a set of features, returns a prediction (forward pass)
    """

    def __init__(self, input_dim=9, output_dim=1, depth=1, inner_width=9, **kwargs): # TODO: confirm **kwargs necessity
        
        super().__init__()
        self.layers = torch.nn.Sequential() # define layers
        # Input layer
        self.layers.add_module("input_layer", torch.nn.Linear(input_dim, inner_width))
         # Hidden layers
        self.layers.add_module("relu1", torch.nn.ReLU())
        for i in range(depth - 1):
            self.layers.add_module(f"hidden_layer{i}", torch.nn.Linear(inner_width, inner_width))
            self.layers.add_module(f"relu{i + 1}",  torch.nn.ReLU())
        # Output layer
        # TODO: output dimension has to be the same as the number of categories
        self.layers.add_module("output_layer",  torch.nn.Linear(inner_width, output_dim))

    def forward(self, X):
        return F.softmax(self.layers(X)) 


def train(model, train_loader, validation_loader, optimizer='Adam', learning_rate='0.001', epochs=10):
    """
        Training function for the Neural Network        
        Parameters:
            - Neural network model (an instance of the NN class)
            - number of epochs for the training
            - the optimizer
        Returns, for the trained model:
            - loss.item()
            - R_squared
            - validation_loss
            - training_time
            - average_inference_latency
    """
    if optimizer == 'Adam': # initialize the optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Currently supporting 'Adam' optimizer only")

    writer = SummaryWriter() # Tensorboard class initialization
    batch_idx = 0 # initalize outside the epochs loop so to create an index for the writer class
    training_time = 0 # initialize time performance indicators
    cumulative_inference_latency = 0
    
    for epoch in range(epochs): # outer loop: epochs

        print("\nEpoch: ", epoch, "/", epochs)
        training_start_time = time.time()

        for batch in train_loader: # inner loop: training
            # print(batch_idx)
            features, labels = batch
            prediction = model(features) # forward step and loss calculation
            loss = F.binary_cross_entropy(prediction, labels)
            loss.backward() # loss differentiation (backward step)
            optimizer.step() # optimization step
            optimizer.zero_grad() # set gradient to zero

            writer.add_scalar('training loss rmse', # TensorFlow writer: add a step
                              np.sqrt(loss.item()), batch_idx)
            batch_idx += 1 # TensorFlow writer: increase the index for next step

        training_stop_time = time.time()
        training_time += training_stop_time - training_start_time # calculate the training time
        print("\nTraining time: ", training_time)

        validation_loss, inference_time, r_squared = validate(model=model, dataset=validation_loader)
        cumulative_inference_latency += inference_time

        writer.add_scalar('validation loss rmse', # TensorFlow writer
                             np.sqrt(validation_loss.item()), epoch) 
    
    #calculate the batch inference latency as an average across all epochs
    average_inference_latency = cumulative_inference_latency/epochs
    print("\nAverage inference latency:", average_inference_latency)

    return np.sqrt(loss.item()), r_squared, np.sqrt(validation_loss.item()), training_time, average_inference_latency


def validate(model, dataset):
    """
        Performs a forward pass and returns loss and inference time
        
        Returns:
            - the loss MSE
            - inference time
    """
    for batch in dataset:
        features, labels = batch
        inference_start_time = time.time()
        prediction = model(features)
        inference_stop_time = time.time() # time taken to perform a forward pass on a batch of features
        inference_time = inference_stop_time - inference_start_time
        validation_loss = F.binary_cross_entropy(prediction, labels)
        #r_squared = r2_score(prediction, labels)
        r_squared = 0
        print("\nValidation loss rmse: ", np.sqrt(validation_loss.item()))
    
    return validation_loss, inference_time, r_squared


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


def save_model(model, config,
               RMSE_loss=None, validation_loss=None, R_squared=None,
               training_duration=None, inference_latency=None,
               folder_path="./neural_networks/regression/"):
    
    current_time = datetime.now()    # Get the current date and time
    folder_name = current_time.strftime('%Y-%m-%d_%H.%M.%S') # Format the current time as a string in the desired format

    model_path = folder_path + folder_name
    os.mkdir(model_path)     # Create a directory with the formatted name
    torch.save(model.state_dict(), model_path+'/model.pt')
    
    hyperparameters_file = '/hyperparameters.json'
    metrics_file = '/metrics.json'

    metrics = {'RMSE_loss': RMSE_loss, 'R_squared': R_squared, 'validation_loss': validation_loss, 'training_duration': training_duration, 'interference_latency': inference_latency}

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
    sum_of_squared_residuals = torch.sum((labels - predictions)**2)  # SSR sum of squared residuals
    total_sum_of_squares = torch.sum((labels - mean_labels)**2)      # SST total sum of squares

    r2 = 1 - (sum_of_squared_residuals / total_sum_of_squares)
    #print("Coefficient of determination: ", r2)

    return r2.item()


def generate_nn_configs(hyperparameters: typing.Dict[str, typing.Iterable]):
    """
        Generates a parameters grid from a dictionary. It uses Cartesian product
        to generate the possible combinations.
        It uses a generator expression to yield each combination.
        Parameters:
            parameters_grid, which is expected to be a dictionary where keys
            represent parameter names (as strings), and values are iterable collections
            (e.g., lists or tuples) containing possible values for those parameters.
        Returns:
            a generator expression to yield each combination.
    """
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


def find_best_nn(grid, train_loader, validation_loader, performance_indicator = "rmse"):
    """
        Parameters:
            - A dictionary containing the model parameters grid
            - training data loader
            - a validation dataset
        Returns:
            - the best model from the grid, saved in a separate folder for each run
    """

    # Generate the parameters grid
    hyperparameters_grid = generate_nn_configs(grid) # generate the grid

    # TODO: if different performance indicators a needed, then expand the "if"
    if performance_indicator == "rmse":
        best_performance = np.inf

    # loop through the grid to find the best model
    for config in hyperparameters_grid:        
        model = NN(**config) # initialize an instance of the NN class with config parameters
        # TODO: it might be those parameters are already in the model.parameters() and not needed at all
        #print(config)
        #print(model)
        #print(model.parameters())
        # perform the model training
        loss, R_squared, validation_loss, training_time, inference_latency = train(
                                                                                model,
                                                                                train_loader=train_loader,
                                                                                validation_loader=validation_loader,
                                                                                optimizer=config["optimizer"],
                                                                                learning_rate=config["learning_rate"],
                                                                                epochs=config["epochs"])
        
        # Compare the model performance with the best. If better, update the values 
        if validation_loss < best_performance:
            best_training_loss = loss
            best_validation_loss = validation_loss
            best_R_squared = R_squared
            best_model = model
            best_model_hyperparameters = config
            best_model_training_time = training_time
            best_model_inference_latency = inference_latency

    save_model(best_model, best_model_hyperparameters,
               RMSE_loss=best_training_loss,
               validation_loss=best_validation_loss,
               R_squared=best_R_squared,
               training_duration=best_model_training_time,
               inference_latency=best_model_inference_latency,
               folder_path="./neural_networks/regression/")
    

if __name__ == "__main__":
    
    # TODO: implement load_airbnb one-hot-encoding dynamically
    dataset_path = "./airbnb-property-listings/tabular_data/clean_tabular_data.csv"
    #dataset_path = "./airbnb-property-listings/tabular_data/clean_tabular_data.csv"

    label = "Category"

    # initialize an instance of the class which creates a PyTorch dataset
    dataset = AirbnbNightlyPriceClassificationDataset(dataset_path=dataset_path, label=label)
    
    train_loader, validation_loader, test_loader = data_loader(dataset, test_batch_size=32, shuffle=True)
    
    grid = {
        "input_dim": [len(dataset[0][0])],
        "inner_width" :[len(dataset[0][0])],
        "learning_rate": [0.01, 0.001],
        "depth": [1, 2],
        "batch size": [32],
        "epochs": [20],
        "optimizer": ['Adam', 'SGD']
        }
    
    find_best_nn(grid=grid, train_loader=train_loader, validation_loader=validation_loader, performance_indicator='rmse')