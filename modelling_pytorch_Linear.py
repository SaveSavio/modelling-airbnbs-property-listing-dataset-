from typing import Any
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tabular_data import database_utils as dbu
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__() # this bit is needed when inheriting so to initialize the parent class methods
        df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data_one-hot-encoding.csv")
        self.features, self.labels = dbu.load_airbnb(df, label="Price_Night", numeric_only=True)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx]).float() # simply gets the idx example and transfor it into a torch object
        label = torch.tensor(self.labels.iloc[idx]).float()
        return (features, label)
    
    def __len__(self):
        return len(self.features)

def data_loader(dataset, train_ratio=0.7, val_ratio=0.15, batch_size=32, shuffle=True):    
    
    dataset_size = len(dataset)
    # Calculate the number of samples for each split
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Use random_split to split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # not so nice to see...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class LinearRegression(torch.nn.Module):
    """
        Defines single linear layer regression in PyTorch
    """
    def __init__(self):
        super().__init__() 
        self.linear_layer = torch.nn.Linear(23, 1)

    def forward(self, features):
        return self.linear_layer(features)


def train(model, epochs = 10):
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    # torch provides model parameters throught the .parameters() method
    # this is inherited from the torch class

    writer = SummaryWriter()
    batch_idx = 0 # initalize outside the epoch

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward() # differentiate the loss
            optimizer.step() # optimization step by
            optimizer.zero_grad() # set the gradient to zero

            writer.add_scalar('loss = RMSE', np.sqrt(loss.item()), batch_idx) # cannot used the batch index because it resets every epoch
            batch_idx += 1

model = LinearRegression()

if __name__ == "__main__":
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_loader, val_loader, test_loader = data_loader(dataset, batch_size=32, shuffle=True)
    train(model, epochs=100)