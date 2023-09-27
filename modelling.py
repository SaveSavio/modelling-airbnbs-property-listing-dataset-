import glob
import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
import typing
# from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from tabular_data import load_airbnb
from typing import Type



# print(f'The number of training examples is {X_train.shape[0]}\n The number of labels is {X_train.shape[1]}')

# # linear regression model using sklearn SGDRegressor
# sgd_regressor = SGDRegressor(max_iter=10^5, random_state=1)
# sgd_regressor.fit(X_train, y_train)
# y_pred = sgd_regressor.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# #print('RMSE = ', np.sqrt(mse))

# r2 = r2_score(y_test, y_pred)
# #print('r2 = ', r2)


def custom_tune_regression_model_hyperparameters(mode_class_obj: Type, parameters_grid: dict,
        X_train, X_validation, X_test, y_train, y_validation, y_test):
    """
        A function designed to tune the regression model hyperparameters.
        Implemented explicitely, i.e. without employing sklearn GridSearchCV
        Paremeters:
            - The model class
            - A dictionary of hyperparameter names mapping to a list of values to be tried
            - The training, validation, and test sets
        Returns:
            - the best model
            - a dictionary of its best hyperparameter values
            - a dictionary of its performance metrics.
    """

    best_hyperparams, best_loss = None, np.inf

    def grid_search(parameters_grid: typing.Dict[str, typing.Iterable]):
        keys, values = zip(*parameters_grid.items())
        yield from (dict(zip(keys, v)) for v in itertools.product(*values))

    for hyperparams in grid_search(parameters_grid):
        model = mode_class_obj(**hyperparams)
        model.fit(X_train, y_train)

    y_validation_pred = model.predict(X_validation)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)

    print(f"H-Params: {hyperparams} Validation loss: {validation_loss}")
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_hyperparams = hyperparams

    y_pred = model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)
    model_performance = {"best hyperparameters": best_hyperparams, "validation_RMSE": test_loss}
    return model_performance


# TO DO: consider other metrics, such as MAE and/or R^2
def tune_regression_model_hyperparameters(mode_class_obj: Type, parameters_grid: dict,
    X_train, X_test, y_train, y_test):
    """
        A function designed to tune the regression model hyperparameters. Employs sklearn GridSearchCV
        Paremeters:
            - The model class
            - A dictionary of hyperparameter names mapping to a list of values to be tried
            - The training, validation, and test sets
        Returns:
            - the best model
            - a dictionary of its best hyperparameter values
            - a dictionary of its performance metrics.
    """

    grid_search = GridSearchCV(mode_class_obj(random_state = 1), parameters_grid)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and the best model
    best_hyperparams = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Train the best model on the test dataset and evaluate performance
    best_model.fit(X_test, y_test)
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) # this is the test loss
    r2 = r2_score(y_test, y_pred) # additional metric
    mae = mean_absolute_error(y_test, y_pred)  # additional metric
    test_loss = rmse
    model_performance = {"best hyperparameters": best_hyperparams, "validation_RMSE": test_loss, "R^2": r2, "MAE": mae}
    print(model_performance)
    return model_performance

def save_model(model, model_filename, folder_path, model_performance):
    """
        Saves a regression model in the desired folder path, alongside its performance indicators
        Parameters:
            - model class
            - model filename 
            - model path
            - dictionary containing the summary of the model perfomance on the dataset
    """
    full_model_path = folder_path + model_filename

    if os.path.isdir(folder_path) == False:
        os.mkdir(folder_path)
    
    joblib.dump(model, full_model_path)
    print(f"Model saved to {model_filename}")
        
    # Write the dictionary to the JSON file
    full_performance_path = full_model_path + '.json'
    with open(full_performance_path, "w") as json_file:
      json.dump(model_performance, json_file)
    

def evaluate_all_models(model_list , parameter_grid_list, X_train, X_test, y_train, y_test):
    # decision trees, random forests, and gradient boosting
    # It's extremely important to apply your tune_regression_model_hyperparameters function
    # to each of these to tune their hyperparameters before evaluating them
    # Save the model, hyperparameters, and metrics in a folder named after the model class.
    # For example, save your best decision tree in a folder called models/regression/decision_tree.
    # rand_forest = RandomForestRegressor()
    # grad_boost = GradientBoostingRegressor()
    for index, model in enumerate(model_list):
        print(model, parameter_grid_list[index])
        model_performance = tune_regression_model_hyperparameters(model, parameter_grid_list[index],
                                                                  X_train, X_test, y_train, y_test)
        save_model(model, model_filename='best_'+model.__name__, folder_path='models/regression/'+model.__name__+'/',
                   model_performance=model_performance)

def find_best_model(search_directory = './models/regression', evaluation_metric='rmse'):
    """
        Finds the best model amongst those in a folder path by comparing their "evaluation metric"
        Returns:
            - loaded model
            - a dictionary of its hyperparameters
            - a dictionary of its performance metrics.
    """
    # Define the file extension you want to search for
    file_extension = '*.json'

    # Use glob to find all JSON files in the specified directory and its subdirectories
    json_files = glob.glob(os.path.join(search_directory, '**', file_extension), recursive=True)

    # Print the list of JSON files found
    min_rmse = np.inf
    for json_file in json_files:
        print(json_file)
        with open(json_file, "r") as json_file:
            data = json.load(json_file)
            if data['validation_RMSE'] < min_rmse:
                min_rmse = data['validation_RMSE']
                model = json_file
                print(json_file)
        return model

if __name__ == "__main__":
    # load the previously cleaned data
    df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    # select labels and features
    features, labels = load_airbnb(df, label="Price_Night", numeric_only=True)
    # rescale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features) # Fit and transform the data
    # split in train, validation, test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.3)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

    model_list = [SGDRegressor, RandomForestRegressor, GradientBoostingRegressor]
    
    parameter_grid_list = [
        {'alpha': [0.0001, 0.001, 0.01, 0.1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'max_iter': [10**4, 10**5, 10**6]}, 
        {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]},
        {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
         'max_depth': [10, 20, 30],
         'min_samples_split': [2, 3, 4],
         'min_samples_leaf': [1, 2, 4]
         }
        ]
    
    evaluate_all_models(model_list, parameter_grid_list, X_train, X_test, y_train, y_test)
    # find_best_model()