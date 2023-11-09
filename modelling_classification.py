import glob
import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
import typing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tabular_data import database_utils as dbu
from typing import Type

def features_scaling(df, columns_to_scale_index, label):
    """
        Scales the features dataframe using sklearn.StandardScaler
        For added flexibility, uses a list of features and the label to determine
        which column is to be scaled.
        Parameters:
            A dataframe
            A list of features_to_scale

        Returns:
            a dataframe whose "features_to_scale" are scaled, with exception of "label"
    """
    # remove the label from the list, there's no need to rescale it
    #columns_to_scale_index.remove(label)
    # create the subset of features that need scaling
    columns_subset = df[columns_to_scale_index]

    scaler = StandardScaler() # features scaling  
    scaled_columns = scaler.fit_transform(columns_subset) # fit and transform the data
    # now substitute the scaled features back in the original dataframe
    df[features_to_scale] = scaled_columns
    #features.head()
    return df


def tune_classification_model_hyperparameters(model_class_obj: Type, parameters_grid: dict,
                                              X_train, X_test, y_train, y_test,
                                              validation_accuracy="accuracy", random_state=1):
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

    grid_search = GridSearchCV(model_class_obj(random_state=random_state),
                               param_grid=parameters_grid,
                               scoring=validation_accuracy, cv=5, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and the best model
    best_hyperparams = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # validation accuracy
    # Train the best model on the test dataset and evaluate performance
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_train)
    validation_accuracy = accuracy_score(y_pred, y_train)

    # Train the best model on the test dataset and evaluate performance
    best_model.fit(X_test, y_test)
    y_pred = best_model.predict(X_test)

    # calculate performance metrics
    test_precision = precision_score(y_test, y_pred, average=None)
    test_recall = recall_score(y_test, y_pred, average=None)
    test_accuracy = accuracy_score(y_test, y_pred)
    #test_f1 = f1_score(y_test, y_pred)

    # change these types from ndarray to list for saving to json
    test_precision = list(test_precision)
    test_recall = list(test_recall)


    # create a dictionary containing: best hyperparameters and performance metrics
    model_info = {"best hyperparameters": best_hyperparams,
                  "Validation Accuracy ": validation_accuracy,
                  "Test Accuracy": test_accuracy,
                  "Test Precision": test_precision,
                  "Test Recall": test_recall}

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
      print(model_info)
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
        model_performance = tune_classification_model_hyperparameters(model, parameter_grid_list[index],
                                                                  X_train, X_test, y_train, y_test)
        
        # define model naming strategy and saving folder path
        model_filename = 'best_'+model.__name__
        task_folder = 'models/classification/'+model.__name__+'/'

        save_model(model, model_filename=model_filename,
                   folder_path=task_folder, model_info=model_performance)


def find_best_model(search_directory = './models/classification'):
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

    max_accuracy = 0
    for json_file in json_files:
            with open(json_file, "r") as file:
                data = json.load(file)
                if data['Accuracy'] > max_accuracy:
                    max_accuracy = data['Accuracy']
                    best_model = json_file[:-4] + 'pkl'
                    best_validation_accuracy = data.get('Validation Accuracy')
                    best_test_accuracy = data.get('Test Accuracy')
                    best_hyperparameters = data.get('best hyperparameters')
    
    # loads the model
    best_model = joblib.load(best_model)
    print("Best model loaded: ", best_model,
          "\nValidation accuracy: ", best_validation_accuracy,
          "\nTest accuracy: ", best_test_accuracy, 
        "\nHyper-parameters: ", best_hyperparameters)
    return best_model, best_performance, best_hyperparameters


if __name__ == "__main__":
    data_path = "./airbnb-property-listings/tabular_data/clean_tabular_data.csv"

    # load the previously cleaned data
    df = pd.read_csv(data_path)
    
    # define labels and features
    label = "Category"
    features, labels = dbu.load_airbnb(df, label=label, numeric_only=True)
    features.head()
    labels.head()
    # create a list of numerical features
    features_to_scale = ['guests', 'beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating',
                            'Accuracy_rating', 'Communication_rating', 'Location_rating',
                            'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms']

    scaled_features = features_scaling(features, features_to_scale, label)

    # split in train, validation, test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    #X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

    model_list = [LogisticRegression, # model 1
                  RandomForestClassifier, # model 2
                  GradientBoostingClassifier, # model 3
                  DecisionTreeClassifier] # model 4

    #model_list = [LogisticRegression] # model 1


    param_grid_list = [
                { # model 1
                'penalty': ['l1', 'l2'],            # Regularization type ('l1' for L1, 'l2' for L2)
                'C': [0.001, 0.01, 0.1, 1.0, 10], # Inverse of regularization strength
                'solver': ['liblinear', 'saga'],    # Solver algorithms (suitable for small to medium datasets)
                'max_iter': [10**3, 10**4]        # Maximum number of iterations for solver convergence]
                },
                { # model 2
                'n_estimators': [10, 100, 500],         # Number of trees in the forest
                'max_depth': [None, 10, 20, 30],      # Maximum depth of the trees
                'min_samples_split': [2, 5, 10],      # Minimum number of samples required to split a node
                'min_samples_leaf': [1, 2, 4],        # Minimum number of samples required at each leaf node
                'class_weight': [None, 'balanced']
                },
                { # model 3
                'n_estimators': [10, 100, 500],       # Number of boosting stages to be used
                'learning_rate': [0.01, 0.1, 0.2],   # Step size shrinkage used to prevent overfitting
                'max_depth': [3, 4, 5],              # Maximum depth of the individual trees
                'subsample': [0.8, 0.9, 1.0]
                },
                { # model 4
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
                }
                ]

    # evaluate all models in the model list according to the parameters in the grid
    # for each model type, save the best
    evaluate_all_models(model_list, param_grid_list, X_train, X_test, y_train, y_test)

    best_model, best_performance, best_hyperparams = find_best_model()