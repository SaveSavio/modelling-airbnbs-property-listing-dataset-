from typing import Any
import pandas as pd
import numpy as np
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

class LinearRegression(torch.nn.Module):
    """
        Defines single linear layer regression in PyTorch
    """
    def __init__(self):
        super().__init__() 
        self.linear_layer = torch.nn.Linear(9, 1)

    def forward(self, features):
        return self.linear_layer(features)


def train(model, epochs = 100):
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # torch provides model parameters throught the .parameters() method
    # this is inherited from the torch class

    writer = SummaryWriter()
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
            #print(loss.item())
            optimizer.step()
            optimizer.zero_grad() # this is necessary because of the behaviour of .backward() not overwriting values

            writer.add_scalar('loss', loss.item(), batch_idx) # cannot used the batch index because it resets every epoch
            batch_idx += 1

model = LinearRegression()

if __name__ == "__main__":
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_loader, val_loader, test_loader = data_loader(dataset, batch_size=32, shuffle=True)
    train(model)