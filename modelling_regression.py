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
from sklearn.tree import DecisionTreeRegressor
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
    # root mean square error
    validation_loss = np.sqrt(mean_squared_error(y_validation, y_validation_pred))  

    print(f"H-Params: {hyperparams} Validation loss: {validation_loss}")
    
    # compares to best loss, if better, and updates best loss and hyperparams
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_hyperparams = hyperparams

    # finally makes a prediction on test data
    y_pred = model.predict(X_test)
    test_loss = np.sqrt(mean_squared_error(y_test, y_pred))
    model_performance = {"best model": model_class_obj,
                         "best hyperparameters": best_hyperparams,
                         "validation_RMSE": validation_loss,
                         "test_RMSE": test_loss}

    return model_performance


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
    columns_to_scale_index.remove(label)
    # create the subset of features that need scaling
    columns_subset = df[columns_to_scale_index]

    scaler = StandardScaler() # features scaling  
    scaled_columns = scaler.fit_transform(columns_subset) # fit and transform the data
    # now substitute the scaled features back in the original dataframe
    df[features_to_scale] = scaled_columns
    #features.head()
    return df


def tune_regression_model_hyperparameters(mode_class_obj: Type, parameters_grid: dict,
    X_train, X_validation, y_train, y_validation, random_state=3):
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
    grid_search = GridSearchCV(mode_class_obj(), parameters_grid, cv=10,
                               scoring='neg_root_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_search.fit(X_train, y_train) # fit the gridsearch on the training set
    
    # cv_results = grid_search.cv_results_ # A dict with all results for the grid search
    # df = pd.DataFrame.from_dict(cv_results) 
    # df.to_csv (r'grid_search_results.csv', index=False, header=True)

    best_hyperparams = grid_search.best_params_ # Parameter setting that gave the best results on the hold out data
    best_model = grid_search.best_estimator_ # estimator which gave highest score (or smallest loss if specified) on the left out data.
    best_rmse = -grid_search.best_score_ # Mean cross-validated score of the best_estimator

    y_val_pred = best_model.predict(X_validation) # predict on the validation set
    validation_rmse = np.sqrt(mean_squared_error(y_validation, y_val_pred))
    validation_r2 = r2_score(y_validation, y_val_pred)
    validation_mae = mean_absolute_error(y_validation, y_val_pred)

    prediction_csv = pd.DataFrame({'y_validation': y_validation, 'y_prediction': y_val_pred})
    prediction_csv.to_csv (r'prediction.csv', index=False, header=True)

    # create a dictionary containing: best hyperparameters and performance metrics
    model_info = {"best hyperparameters": best_hyperparams,
                  "training_RMSE": best_rmse,
                  "validation_RMSE": validation_rmse,
                  "validation_R^2": validation_r2,
                  "validation_MAE": validation_mae}
    print("model info: ", model_info)

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
    print("full_model_path: ")
    print(full_model_path)

    if os.path.isdir(folder_path) == False:
        os.mkdir(folder_path)
    
    joblib.dump(model, full_model_path + '.pkl')
    print(f"Model saved to {model_filename}")
        
    # Write the dictionary to the JSON file
    full_performance_path = full_model_path + '.json'
    with open(full_performance_path, "w") as json_file:
      json.dump(model_info, json_file)
    

def evaluate_all_models(model_list: list , parameter_grid_list: list, X_train, X_validation, y_train, y_validation, directory):
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
        print('Estimator: ', model, '\nHyperparameters grid list: ', parameter_grid_list[index])

        model_performance = tune_regression_model_hyperparameters(model, parameter_grid_list[index],
                                                                  X_train, X_validation, y_train, y_validation)
        
        # define model naming strategy and saving folder path
        model_filename = 'best_' + model.__name__
        task_folder = directory + model.__name__+'/'

        save_model(model, model_filename=model_filename,
                   folder_path=task_folder, model_info=model_performance)


def find_best_model(search_directory = './models/regression'):
    """
        Finds the best model amongst those in a folder path by comparing their evaluation metric.
        The set evaluation metric is the RMSE on the validation dataset.
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
                    train_rmse = data.get('training_RMSE')
                    validation_rmse = data.get('validation_RMSE')
                    validation_r2 = data.get('validation_R^2')
                    validation_mae = data.get('validation_MAE')
                    best_performance = [validation_rmse, train_rmse]
                    best_hyperparameters = data.get('best hyperparameters')
    
    # loads the model
    best_model = joblib.load(best_model)
    print("Best model loaded: ", best_model,
          "\nHyper-parameters: ", best_hyperparameters,
          "\ntrain_RMSE: ", train_rmse,
          "\nvalidation_RMSE: ", validation_rmse,
          "\nValidation_R^2: ", validation_r2,
          "\nValidation_MAE:", validation_mae)
    return best_model, best_performance, best_hyperparameters


if __name__ == "__main__":

    data_paths = ["./airbnb-property-listings/tabular_data/clean_tabular_data_one-hot-encoding.csv",
                 "./airbnb-property-listings/tabular_data/clean_tabular_data_one-hot-encoding_remove_price_night_outliers.csv",
                 "./airbnb-property-listings/tabular_data/clean_tabular_data_one-hot-encoding_price_night_outliers_only.csv"]
    
    directories = ['./models/regression/full_dataset',
                 './models/regression/outliers_removed',
                 './models/regression/outliers']

    for idx in range(len(data_paths)):
        data_path = data_paths[idx]
        directory = directories[idx]

        df = pd.read_csv(data_path) # load the previously cleaned data

        label = 'Price_Night' # define labels and, subsequently, features

        features, labels = dbu.load_airbnb(df, label=label, numeric_only=True)
        
        features_to_scale = [ # create a list of numerical features

            'guests', 'beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating',
                                'Accuracy_rating', 'Communication_rating', 'Location_rating',
                                'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms']
        
        scaled_features = features_scaling(features, features_to_scale, label) # apply features scaling

        X_train, X_validation, y_train, y_validation = train_test_split( # split data in train and validation sets
            scaled_features, labels, test_size=0.3)

        # list of models to be used for the regression
        model_list = [SGDRegressor, # model 1
                    DecisionTreeRegressor, # model 2
                    RandomForestRegressor, # model 3
                    GradientBoostingRegressor] # model 4
        
        parameter_grid_list = [ # grid list for model optimization, one dict for each model

            {
            'alpha': [0.001, 0.01, 0.1], # model 1
            'penalty': ['l2', 'l1', 'elasticnet'],
            'loss': ['squared_error'],
            'learning_rate':['constant', 'adaptive'],
            'eta0': [0.001, 0.01],
            'max_iter': [10**5]
            },
            {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'ccp_alpha': [0.0, 0.1, 0.2],
            },
            {
            'n_estimators': [50, 100, 200], # model 3
            'max_depth': [None, 10, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            },
            {
            'n_estimators': [10, 50, 100, 200], # model 4
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [None, 10, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            }
            ]
        
        evaluate_all_models( # evaluate all models for grid, save the best model for each type
            model_list, parameter_grid_list,
            X_train, X_validation, y_train, y_validation,
            directory=directory)
        
        # find the best overall model for regression
        # best_model, best_performance, best_hyperparams = find_best_model(search_directory=directory)