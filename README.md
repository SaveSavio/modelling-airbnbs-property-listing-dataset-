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

The code for preparation of tabular data is placed inside <u>tabular_data.py</u>:<br> that, when called, performs the cleaning if called and saves the clean data in clean_tabular_data.csv.

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
<u>NOTE:</u>:<br>
This framework works only for numerical variables. If categorical variables are to be used, one-hot-encoding should be implemented.

The framework build in <u>modelling.py</u>:<br> allows to systematically compare the performance of regression models.

The main function is
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

The other functions in the file are:
```python
def evaluate_all_models(model_list: list , parameter_grid_list: list, X_train, X_test, y_train, y_test):
def save_model(model, model_filename: str, folder_path: str, model_info: dict):
def find_best_model(search_directory = './models/regression'):
```
They respectively:
- evaluate the models provided in a list, alongside a list of parameters grids (one for each model)
- save the best model for each type (or model class, e.g. SGDRegressor, RandomForestRegressor etc)
- find the best overall model (i.e. among all the types of models)

def custom_tune_regression_model_hyperparameters(model_class_obj: Type, parameters_grid: dict,

## Classification models
The same framework as for the regression models, has been build for a classification type example.
The file <u>modelling_classification.py</u>:<br> contains the same functions and architeture as the regression case, 
but adapted to classification problems.


