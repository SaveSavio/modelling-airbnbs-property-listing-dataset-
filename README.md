# Modelling Airbnb's property listing dataset 

<u>Motivation</u>:<br>
To build a framework to systematically trains, tunes, and evaluates models on several tasks that are tackled by the Airbnb team.

## Installation
Ensure the following dependencies are installed or run the following commands using pip package installer (https://pypi.org/project/pip/):

```python
    pip install # following packages:
    glob
    itertools
    joblib
    json
    numpy
    os
    pandas
    typing
    scikit-learn
    torch
    yaml
    datetime
```
The remaining packages are part of Python's standard library, so no additional installation is needed.

## Data Preparation
The AirBnb listings data comes in form of:
- tabular data ('AirBnbData.csv' file)
- images (a series of '.png' files, one folder for each listing)

The tabular dataset has the following columns:

- ID: Unique identifier for the listing
- Category: The category of the listing
- Title: The title of the listing
- Description: The description of the listing
- Amenities: The available amenities of the listing
- Location: The location of the listing
- guests: The number of guests that can be accommodated in the listing
- beds: The number of available beds in the listing
- bathrooms: The number of bathrooms in the listing
- Price_Night: The price per night of the listing
- Cleanliness_rate: The cleanliness rating of the listing
- Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
- Location_rate: The rating of the location of the listing
- Check-in_rate: The rating of check-in process given by the host
- Value_rate: The rating of value given by the host
- amenities_count: The number of amenities in the listing
- url: The URL of the listing
- bedrooms: The number of bedrooms in the listing

The code for preparation of tabular data is placed inside <u>tabular_data.py</u>. This file, when called, performs the cleaning if called and saves the clean data in clean_tabular_data.csv.

At first, the 'AirBnbData.csv' file into a pandas dataframe
```python
df = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")
```
then calling on it:
```python
def clean_tabular_data(df)
```

This function takes a the listings in form pandas dataframe as argument and cleanes it by calling three separate functions:
```python
def remove_rows_with_missing_ratings(df)
def combine_description_strings(df)
def set_default_feature_values(df)
```
They, in turn:
- Remove the rows with missing values in each of the rating columns
- Combine the list items into the same string
- Replace the empty rows in the colums "guests", "beds", "bathrooms", "bedrooms" with a default value equal to 1

The function
```python
def load_airbnb(df, label="label", numeric_only=False):
```
splits the dataframe into features and labels in order to prepare it for Machine Learning.
Furthermore, if required, can remove all non-numeric values from the dataset.

## Regression models
<u>NOTE:</u><br>
This framework works only for numerical variables. If categorical variables are to be used, one-hot-encoding should be performed.

The framework build in <u>modelling.py</u> allows to systematically compare the performance of regression models.

The main function is:
```python
def tune_regression_model_hyperparameters(mode_class_obj: Type, parameters_grid: dict,
    X_train, X_test, y_train, y_test, random_state = 1)
```
Which is designed to tune the regression model hyperparameters by using sklearn GridSearchCV.

It takes the following paremeters:
- The model class
- A dictionary of hyperparameter names mapping to a list of values to be tried
- The training, validation, and test sets

and returns:
- the best model
- a dictionary of its best hyperparameter values
- a dictionary of its performance metrics.

```python
def custom_tune_regression_model_hyperparameters(model_class_obj: Type, parameters_grid: dict,
        X_train, X_validation, X_test, y_train, y_validation, y_test):
```
performs the same task but explicitly performing the model tuning without calling GridSearchCV.

The other functions in the file are:
```python
def evaluate_all_models(model_list: list , parameter_grid_list: list, X_train, X_test, y_train, y_test):
```
 that evaluates the models provided in a list, alongside a list of parameters grids (one for each model)

```python
def save_model(model, model_filename: str, folder_path: str, model_info: dict):
```
saves the best model for each type (or model class, e.g. SGDRegressor, RandomForestRegressor etc)

```python
def find_best_model(search_directory = './models/regression'):
```
finds the best overall model (i.e. among all the types of models)

## Classification models
The same framework as above has been build for a classification type example.
The file <u>modelling_classification.py</u> contains the same functions and architeture as the regression case, 
but adapted to classification problems.

The validation will accept a validation_accuracy parameter that determines the validation metric
and that is passed to GridSearchCV.
```python
def tune_classification_model_hyperparameters(model_class_obj: Type, parameters_grid: dict,
                                              X_train, X_test, y_train, y_test,
                                              validation_accuracy="accuracy", random_state = 1):
```
## Neural Network Regression

The file <u>modelling_pytorch_Linear.py</u> is an implementation of Linear Regression using PyTorch framework.
This "shallow" model is just an intermediate step to the build of a deep learning model.

The file <u>modelling_NN.py</u> contains the framework to make predictions on the AirBnB database
with Neural Networks with PyTorch.

## Neural Network Workflow

We will follow the process inside the block:
```python
if __name__ == "__main__":
```

### Data Creation
At first, set the path to the cleaned tabular data
```python
dataset_path = "./airbnb-property-listings/tabular_data/clean_tabular_data.csv"
```
and decide which label we want to predict
```python
label = "Price_Night"
```
Then initialize an instance of the PyTorch dataset, which creates a 
Tensor for the features and an array for the labels
```python
dataset = AirbnbNightlyPriceRegressionDataset(dataset_path=dataset_path, label=label)
```
The dataset is then passed to the DataLoader which is embedded in the data_loader function which takes the dataset in and returns it split in training, test and validation sets.
```python
train_loader, validation_loader, test_loader = data_loader(dataset, batch_size=32, shuffle=True)
```

### Modelling and model tuning
We will now describe the most general case of usage, that in which we set a grid for the hyperparameters tuning. Here's an example of the grid:
```python
grid = {
    "learning_rate": [0.01, 0.001],
    "depth": [2, 3],
    "hidden layer width": [8, 16],
    "batch size": [16, 32],
    "epochs": [10, 20]
    }
```
Once the grid is set, it can be used as a parameter in the function
```python
def find_best_nn(grid, performance_indicator = "rmse"):
```
that sets the workflow for the modelling framework. It calls
```python
def generate_nn_configs()
```
that uses a generator expression to yield all of the possible combinations in the grid. For each of the possible hyperparameters sets, it initialized an instance of the neural network class
```python
# initialize an instance of the NN class
class NN(torch.nn.Module)
# with the grid parameters
model = NN(**config)
```
The NN class inherits from torch.nn.Module and defines a configurable, fully connected Neural Network. It uses ReLU activation function and has the following configurable parameters:
- input layer dimension
- model depth
- hidden layers width
- output layer dimension

```python
def find_best_nn(grid, performance_indicator = "rmse"):
```
then trains each NN, finds the best performing based on the performance indicator and saves it in a folder alonside two dictionaries:
- hyperparameters.json
- metrics.json
The first contains the best model hyperparameters whilst the second contains all the metrics of the model. An example below:
```python
{"RMSE_loss": 32656.9140625,
"R_squared": -14.05229663848877,
"validation_loss": 15421.5537109375, 
"training_duration": 4.939751386642456, 
"interference_latency": 0.00010931015014648438}
```

The training function takes in the model class, the number of epoch, the optimizer (currently only supports 'Adam').

```python
def train(model, epochs = 10, optimizer='Adam', **kwargs):
```
It also creates a Tensorflow instance that allows to track the model training performance  on the tuning (every batch) and validation (every epoch) sets.
This function returns the performance paramters that will be stores in the
metrics.json file
```python
return loss.item(), R_squared, validation_loss.item(), training_time average_inference_latency
```

## Next steps
This framework is aimed to describe the workflow of Machine Learning. It uses an Occam Razor approach hence starting with simplest Linear Regression model then evolving to Neural Networks.

Main focus is on the training and tuning of models.

1) It should be noticed that no deep EDA was performed on the data. That would be a necessary step to better understand the dataset

1) Clearly some improvements to the accuracy could be made by treating outliers and reducing the skewness of some data.

A deeper use of the provided information:

3) A large chunk of information is discarded by not using the categorical variables. In fact, when predicting the price x night, the listing category would have been useful.

4) A multimodal system that employs the information in the pictures to increase the prediction accuracy

5) A multimodal system that employs the information in the text description