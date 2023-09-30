import glob
import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
import typing
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tabular_data import load_airbnb
from typing import Type


def tune_classification_model_hyperparameters(model_class_obj: Type, parameters_grid: dict,
                                              X_train, X_test, y_train, y_test,
                                              validation_accuracy="accuracy", random_state = 1):
    """
        A function designed to tune the regression model hyperparameters. Uses sklearn GridSearchCV.
        Paremeters:
            - The model class
            - A dictionary of hyperparameter names mapping to a list of values to be tried
            - The training, validation, and test sets
        Returns:
            - the best model
            - a dictionary of its best hyperparameter values
            - a dictionary of its performance metrics.
    """

    grid_search = GridSearchCV(model_class_obj(random_state=random_state), param_grid=parameters_grid,
                               scoring=validation_accuracy, cv=5, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and the best model
    best_hyperparams = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Train the best model on the test dataset and evaluate performance
    best_model.fit(X_test, y_test)
    y_pred = best_model.predict(X_test)
    # calculate performance metrics
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # create a dictionary containing: best hyperparameters and performance metrics
    model_info = {"best hyperparameters": best_hyperparams, "Precision": precision,
                  "Recall": recall, "Accuracy": accuracy, "Confusion Matrix": conf_matrix}

    #print(f"Accuracy: {accuracy:.2f}")
    #print("Confusion Matrix:\n", conf_matrix)
    print(model_class_obj)
    print(model_info)
    return model_info


def save_model(model, model_filename: str, folder_path: str, model_info: dict):
    """
        Saves a regression model in the desired folder path, alongside its performance indicators
        Parameters:
            - model class
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
    # decision trees, random forests, and gradient boosting
    # It's extremely important to apply your tune_regression_model_hyperparameters function
    # to each of these to tune their hyperparameters before evaluating them
    # Save the model, hyperparameters, and metrics in a folder named after the model class.
    # For example, save your best decision tree in a folder called models/regression/decision_tree.
    # rand_forest = RandomForestRegressor()
    # grad_boost = GradientBoostingRegressor()
    for index, model in enumerate(model_list):
        print(model, parameter_grid_list[index])
        model_performance = tune_classification_model_hyperparameters(model, parameter_grid_list[index],
                                                                  X_train, X_test, y_train, y_test)
        save_model(model, model_filename='best_'+model.__name__, folder_path='models/classification/logistic_regression/'+model.__name__+'/',
                   model_performance=model_performance)


def find_best_model(search_directory = './models/classification/logistic_regression'):
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
    # load the previously cleaned data
    df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    # select labels and features
    features, labels = load_airbnb(df, label="Category", numeric_only=True)
    # rescale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features) # Fit and transform the data
    # split in train, validation, test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.3)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

    model_list = LogisticRegression
    
    param_grid = {'penalty': ['l2', 'l1', 'elasticnet', None],
                  'max_iter': [10**2, 10**3, 10**4]}

    model_info = tune_classification_model_hyperparameters(model_list, param_grid,
                                                           X_train, X_test, y_train, y_test,
                                                           validation_accuracy="accuracy", random_state = 1)
    print(model_info)

    # parameter_grid_list = [
    #     {'alpha': [0.0001, 0.001, 0.01, 0.1],
    #     'penalty': ['l2', 'l1', 'elasticnet'],
    #     'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
    #     'max_iter': [10**4, 10**5, 10**6]}, 
    #     {
    #     'n_estimators': [10, 50, 100],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]},
    #     {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #      'max_depth': [10, 20, 30],
    #      'min_samples_split': [2, 3, 4],
    #      'min_samples_leaf': [1, 2, 4]
    #      }
    #     ]
    
    # evaluate_all_models(model_list, parameter_grid_list, X_train, X_test, y_train, y_test)
    # best_model, best_performance, best_hyperparams = find_best_model()