import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tabular_data import load_airbnb

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
        self.features, self.labels = load_airbnb(df, label="Price_Night", numeric_only=True)
    
    def __getitem__(self, index):
        features = torch.tensor(self.features[index])
        label = torch.tensor(self.label[index])
        return (features, label)
    
    def __len__(self):
        return len(self.features)
    
    #DataLoader(dataset=, batch_size=, shuffle=) 
data = AirbnbNightlyPriceRegressionDataset()
    # Define the dataset size
dataset_size = len(data)

# Define the split ratios
train_ratio = 0.7  # 70% for training
val_ratio = 0.15   # 15% for validation
test_ratio = 0.15  # 15% for testing

# Calculate the number of samples for each split
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    data, [train_size, val_size, test_size]
)

# Create data loaders for each split
batch_size = 64  # You can adjust this value based on your needs

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
