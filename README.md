# Modelling Airbnb's property listing dataset 

<u>Motivation</u>:<br>
To build a framework to systematically trains, tunes, and evaluates models on several tasks that are tackled by the Airbnb team.

## Installation
Ensure the following dependencies are installed or run the following commands using pip package installer (https://pypi.org/project/pip/):
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - glob2
  - joblib
  - pytorch
  - pyyaml

Best option is to create a fresh conda environment using environment.yaml:
```python
conda env create -f environment.yaml
```
## Exploratory Data Analysis
On the data, cleaned according to the project requirements, a simple EDA was performed so to understand the key features in the distribution of data. The main purposes are:
1) to better understand the dataset and stimulate curiosity
1) to improve predictions by either regression, classification or neural networks.

The results can be read directly on the Jupyter Notebook <u>exploratory_data_analysis.ipynb</u> available at this link:
[Exploratory Data Analysis](./exploratory_data_analysis.ipynb)

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

Data is read into a Pandas dataframe, then the following actions are performed:

- Remove the rows with missing values in each of the rating columns
- Combine the list items into the same string
- Replace the empty rows in the colums "guests", "beds", "bathrooms", "bedrooms" with a default value equal to 1
- Split the dataframe into features and labels
- one-hot-encode the "Category" variable
- one-hot-encode the "Location" variable after extracting the Country and grouping them by Geographical Area
- (optional) remove all non-numeric values from the dataset

## Regression models
<u>NOTE:</u><br>
The framework code is contained in <u>modelling_regression.py</u> allows to systematically compare the performance of regression models.

The main function is tune_regression_model_hyperparameters which is designed to tune the regression model hyperparameters by using sklearn GridSearchCV with K-fold Cross-Validation.
As expected, this approach is computationally expensive and might the right solution for training of large models.
As an advantage, it makes the best use of data as it does not require a "validation" set for the hyperparameters.

It takes the following paremeters:
- The model class
- A dictionary of hyperparameter names mapping to a list of values to be tried
- The training, validation, and test sets

and returns:
- the best model
- a dictionary of its best hyperparameter values
- a dictionary of its performance metrics.

Optionally, it is possible to use a custom_tune_regression_model_hyperparameters that performs the same task but explicitly performing the model tuning without calling GridSearchCV.

### Regression models: full dataset performance
We have tested the capability of different regressors in predicting the "Price_Night" feature. For more information on the feature, please refer to the Exploratory Data Analysis Jupyter notebook.

Four estimators from the sklearn libraries were tested.
```python
model_list = [SGDRegressor, # model 1
                  DecisionTreeRegressor, # model 2
                  RandomForestRegressor, # model 3
                  GradientBoostingRegressor] # model 4
```
For each estimator, a hyperparameters grid was set and evaluated with a "brute force" method. The best models are saved in the repository folder structure, under the regression folder.

1) None of the models is affected by overfitting. In fact, the RMSE on the validation set is not smaller than the one calculated on the test set.
1) Ranked from best to worst, based on the score on the test set:


| Estimator | Training RMSE | Validation RMSE | Validation R^2 | Validation MAE |
|----------|----------|----------|----------|----------|
| SDGRegressor | 97.01 | 94.92 | 0.41 | 61.31 |
| DecisionTreeRegressor | 100.29 | 107.97 | 0.24 | 69.33 |
| RandomForestRegressor | 92.51 | 98.52 | 0.37 | 62.83 |
| GradientBoostingRegressor | 94.43 | 111.40 | 0.19 |  66.18 |

- The linear SGDRegressor is the baseline for our evaluation:
  - has a validation RMSE of 97$
  - R^2 is 0.41 meaning that only 41% of the Price/Night variance is explained by this model.

  "best hyperparameters": {"alpha": 0.1, "eta0": 0.001, "learning_rate": "adaptive", "loss": "squared_error", "max_iter": 100000, "penalty": "l2"}

- The other 3 models performance is close, with RandomForestRegressor showing marginally the best performance:
  - validation RMSE 92$
  - R^2 of 0.37.

It appears the regression models performance is limited by the amount of information provided by the numerical data.
The R^2 is always below 0.5, meaning our models have limited ability to explain the variability in the label.

### Regression models: outliers
![Boxplot](/price_night_boxplot.png)

The plot above showes the data are quite right skewed. Quite a few listings above the 3rd quartile. In order to improve the RMSE and yet use the numerical data only, we have split the data into a dataset without "Price_Night" outliers. The strategy consists in removing the data above the 90th-percentile (89 examples in total).

A dataset containing only the outliers is created as well. We report the results for the SDG regressor (baseline) and the best performer RandomForestRegressor.

| Estimator | Training RMSE | Validation RMSE | Test R^2 | Test MAE |
|----------|----------|----------|----------|----------|
| SDGRegressor w/o outliers| 49.61 | 44.34 | 0.29 | 35.33 |
| SDGRegressor outliers only | 173.66 | 184.83 | 0.1673 | 127.81 |
| RandomForestRegressor w/o outliers | 49.20 | 45.10 | 0.26 | 35.97 |
| RandomForestRegressor outliers only | 165.67 | 180.15 | 0.21 | 128.79 |

As expected, the RMSE is reduced when removing a portion of the outliers from the dataset.
It is debatable whether this approach can be useful or not in the real world.

## Classification models
A framework analogous to regression - contained in <u>modelling_classification.py</u> - has been developed for a classification scenario.

### Classification models: performance
Four estimators from the sklearn libraries were tested:
```python
model_list = [LogisticRegression, # model 1
                  RandomForestClassifier, # model 2
                  GradientBoostingClassifier, # model 3
                  DecisionTreeClassifier] # model 4
```
The chosen Key Performance Indicator is 'Accuracy':
$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \
$$

Results in the table below
| Estimator | Validation Accuracy | Test Accuracy | 
|----------|----------|----------|
| LogisticRegression | 0.43 | 0.36 |
| RandomForestClassifier | 0.79 | 0.84 |
| GradientBoostingClassifier | 0.76 | 0.92 |
| DecisionTreeClassifier | 0.59 | 0.60 |

The Logistic Regressor is our baseline model for simplicity and for performance. It also shows a certain a tendency to overfit. The Gradient Boosting and the Random Forest regressor outperform the other model giving an accuracy close to 0.9.

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