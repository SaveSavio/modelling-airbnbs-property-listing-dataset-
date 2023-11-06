import glob
import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
import typing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from tabular_data import database_utils as dbu
from typing import Type

def grid_generator(parameters_grid: typing.Dict[str, typing.Iterable]):
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
    
    # unpacks the keys and values from the parameters_grid dictionary. 
    keys, values = zip(*parameters_grid.items())

    # generates the Cartesian product of the iterable collections in values. 
    # dict(zip(keys, v)) is used to create a dictionary for each combination
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


def custom_tune_regression_model_hyperparameters(model_class_obj: Type, parameters_grid: dict,
        X_train, X_validation, X_test, y_train, y_validation, y_test):
    """
        Tunes the regression model hyperparameters.
        Implemented explicitely, i.e. without employing sklearn GridSearchCV
        Paremeters:
            - The model class (and NOT an instance of the class)
            - A dictionary of hyperparameter names mapping to a list of values to be tried
            - The training, validation, and test sets
        Returns:
            - the best model (amongst those in the parameters grid)
            - a dictionary of its best hyperparameter values
            - a dictionary of its performance metrics.
    """

    # initialize performance indicator variable
    best_hyperparams, best_loss = None, np.inf

    # recursively fits the model on the training data
    for hyperparams in grid_generator(parameters_grid):
        model = model_class_obj(**hyperparams)
        model.fit(X_train, y_train)

    # predicts on the validation dataset and calculates the loss
    y_validation_pred = model.predict(X_validation)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)

    print(f"H-Params: {hyperparams} Validation loss: {validation_loss}")
    
    # compares to best loss, if better, and updates best loss and hyperparams
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_hyperparams = hyperparams

    # finally makes a prediction on test data
    y_pred = model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)
    model_performance = {"best model": model_class_obj,
                         "best hyperparameters": best_hyperparams,
                         "validation_RMSE": validation_loss,
                         "test_RMSE": test_loss}

    return model_performance


def tune_regression_model_hyperparameters(mode_class_obj: Type, parameters_grid: dict,
    X_train, X_validation, y_train, y_validation, random_state = 1):
    """
        A function designed to tune the regression model hyperparameters. Uses sklearn GridSearchCV.
        Paremeters:
            - The model class
            - A dictionary of hyperparameter names mapping to a list of values to be tried
            - The training and validation datasets
        Returns:
            - the best model
            - a dictionary of its best hyperparameter values
            - a dictionary of its performance metrics.
    """
    grid_search = GridSearchCV(mode_class_obj(random_state=random_state), parameters_grid, cv=5)
    grid_search.fit(X_train, y_train) # grid search on the training set

    # Get the best hyperparameters and the best model
    best_hyperparams = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # fit and predict on the validation set
    best_model.fit(X_validation, y_validation)
    y_pred = best_model.predict(X_validation)

    rmse = np.sqrt(mean_squared_error(y_validation, y_pred)) # validation loss
    r2 = r2_score(y_validation, y_pred) # additional metrics
    mae = mean_absolute_error(y_validation, y_pred)

    # create a dictionary containing: best hyperparameters and performance metrics
    model_info = {"best hyperparameters": best_hyperparams, "validation_RMSE": rmse, "R^2": r2, "MAE": mae}
    #print(model_info)

    return model_info


def save_model(model, model_filename: str, folder_path: str, model_info: dict):
    """
        Saves a regression model in the desired folder path, alongside its performance indicators
        Parameters:
            - model class object (not an instance of the class)
            - model filename 
            - folder path
            - dictionary containing the summary of the model perfomance on the dataset
    """

    full_model_path = folder_path + model_filename

    if os.path.isdir(folder_path) == False:
        os.mkdir(folder_path)
    
    joblib.dump(model, full_model_path + '.pkl')
    print(f"Model saved to {model_filename}")
        
    # Write the dictionary to the JSON file
    full_performance_path = full_model_path + '.json'
    with open(full_performance_path, "w") as json_file:
      json.dump(model_info, json_file)
    

def evaluate_all_models(model_list: list , parameter_grid_list: list, X_train, X_test, y_train, y_test):
    """
        Evaluates all models in the model list.
        Each model is evaluated according to a grid list by tune_regression_model_hyperparameters function
        Parameters:
            a list of model classes (and NOT instances of a class)
            a grid of model parameters
            test and training datasets (features and labels)
        Returns:
            saves the best model for each class in a folder
    """
    for index, model in enumerate(model_list):
        print(model, parameter_grid_list[index])
        model_performance = tune_regression_model_hyperparameters(model, parameter_grid_list[index],
                                                                  X_train, X_validation, y_train, y_validation)
        
        # define model naming strategy and saving folder path
        model_filename = 'best_'+model.__name__
        task_folder = 'models/regression/'+model.__name__+'/'

        save_model(model, model_filename=model_filename,
                   folder_path=task_folder, model_info=model_performance)


def find_best_model(search_directory = './models/regression'):
    """
        Finds the best model amongst those in a folder path by comparing their rmse (evaluation metric)
        Returns:
            - loaded model
            - a dictionary of its hyperparameters
            - a dictionary of its performance metrics.
    """
    
    # Define the file extension you want to search for
    file_extension = '*.json'

    # Use glob to find all JSON files in the specified directory and its subdirectories
    json_files = glob.glob(os.path.join(search_directory, '**', file_extension), recursive=True)

    min_rmse = np.inf
    for json_file in json_files:
            with open(json_file, "r") as file:
                data = json.load(file)
                if data['validation_RMSE'] < min_rmse:
                    min_rmse = data['validation_RMSE']
                    best_model = json_file[:-4] + 'pkl'
                    best_performance = data.get('validation_RMSE')
                    best_hyperparameters = data.get('best hyperparameters')
    
    # loads the model
    best_model = joblib.load(best_model)
    print("Best model loaded: ", best_model, "\nValidation RMSE: ", best_performance, 
        "\nHyper-parameters: ", best_hyperparameters)
    return best_model, best_performance, best_hyperparameters


if __name__ == "__main__":
    #data_path = "./airbnb-property-listings/tabular_data/clean_tabular_data.csv"
    data_path = "./airbnb-property-listings/tabular_data/clean_tabular_data_transformed.csv"

    
    # load the previously cleaned data
    df = pd.read_csv(data_path)

    # define labels and features
    label = 'Price_Night'
    features, labels = dbu.load_airbnb(df, label=label, numeric_only=True)
    # create a list of numerical features
    features_to_scale = ['guests', 'beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating',
                         'Accuracy_rating', 'Communication_rating', 'Location_rating',
                         'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'] 
    # remove the label from the list, there's no need to rescale it
    features_to_scale.remove(label)

    # create the subset of features that need scaling
    features_subset = features[features_to_scale]

    scaler = StandardScaler() # features scaling  
    scaled_features = scaler.fit_transform(features_subset) # fit and transform the data
     # now substitute the scaled features back in the original dataframe
    features[features_to_scale] = scaled_features
    features.head()

    # split in train, validation, test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.3)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # list of models to be used for the regression
    model_list = [SGDRegressor, # model 1
                  RandomForestRegressor, # model 2
                  GradientBoostingRegressor] # model 3
    
    # grid list of dictonaries for model optimization. Each dictionary for the corresponding model
    parameter_grid_list = [
        {'alpha': [0.001, 0.01, 0.1], # model 1
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'max_iter': [10**4, 10**5,  10**6]}, 
        {
        'n_estimators': [10, 50, 100], # model 2
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]},
        {
        'n_estimators': [50, 100, 200], # model 3
        'learning_rate': [0.01, 0.1, 0.2],
         'max_depth': [10, 20, 30],
         'min_samples_split': [2, 3, 4],
         'min_samples_leaf': [1, 2, 4]
         }
        ]
    
    # evaluate all models in the model list according to the parameters in the grid
    # for each model type, save the best
    evaluate_all_models(model_list, parameter_grid_list, X_train, X_test, y_train, y_test)
    
    # find the best overall model for regression
    best_model, best_performance, best_hyperparams = find_best_model(search_directory = './models/regression/')