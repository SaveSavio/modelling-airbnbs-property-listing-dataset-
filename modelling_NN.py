from typing import Any
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tabular_data import load_airbnb
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml


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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

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


def train(model, epochs = 100, optimizer='Adam', **kwargs): ## TODO: consider if we have to pass train_loader to it
    
    if optimizer == 'Adam': # TODO: should implement alternative optimizers
        # torch provides model parameters throught the .parameters() method
        # this method is inherited from the torch class
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    writer = SummaryWriter() # initialize an instance of this class for Tensorboard

    batch_idx = 0 # initalize outside the epoch

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            # now we need to take an optimization step by
            loss.backward() # differentiate the loss
            # mind you it does not overwrite but add to the gradient
            print(loss)
            # TODO: optimization step
            #print(loss.item())
            optimizer.step()
            optimizer.zero_grad() # this is necessary because of the behaviour of .backward() not overwriting values

            writer.add_scalar('loss', loss.item(), batch_idx) # cannot used the batch index because it resets every epoch
            batch_idx += 1


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

# Adapt your function called save_model so that it detects whether the model is a PyTorch module,
# and if so, saves the torch model in a file called model.pt, its hyperparameters in a file called hyperparameters.json
# and its performance metrics in a file called metrics.json.

# Your metrics should include:

#     The RMSE loss of your model under a key called RMSE_loss for training, validation, and test sets
#     The R^2 score of your model under a key called R_squared for training, validation, and test sets
#     The time taken to train the model under a key called training_duration
#     The average time taken to make a prediction under a key called inference_latency

# Every time you train a model, create a new folder whose name is the current date and time.

# So, for example, a model trained on the 1st of January at 08:00:00 would be saved in a folder called models/neural_networks/regression/2018-01-01_08:00:00.

def save_model(model):
    torch.save(model.state_dict(), 'model.pt')


if __name__ == "__main__":
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_loader, val_loader, test_loader = data_loader(dataset, batch_size=32, shuffle=True)
    config = get_nn_config()
    model = NN(**config)
    train(model, **config)

    #state_dict = model.state_dict()
    #print(state_dict)


    #state_dict = torch.load('model.pt')
    #new_model = NN()
    #new_model.load_state_dict(state_dict)