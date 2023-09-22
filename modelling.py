import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
# TO DO: consider if feature scaling is necessary
from sklearn.preprocessing import StandardScaler
from tabular_data import load_airbnb
from typing import Type


df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
features, labels = load_airbnb(df, label="Price_Night", numeric_only=True)

print(features.columns)
print(labels)

# Rescale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features) # Fit and transform the data

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, labels, test_size=0.3)

X_validation, X_test, y_validation, y_test =train_test_split(
    X_test, y_test, test_size=0.5)

print(f'The number of training examples is {X_train.shape[0]}\n The number of labels is {X_train.shape[1]}')

# create a linear regression model using sklear SGDRegressor
sgd_regressor = SGDRegressor(max_iter=10^5, random_state=1)
sgd_regressor.fit(X_train, y_train)
y_pred = sgd_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('RMSE = ', np.sqrt(mse))

r2 = r2_score(y_test, y_pred)
print('r2 = ', r2)

# import itertools
# import typing


def custom_tune_regression_model_hyperparameters(class_obj: Type, grid: dict, X_train, X_validation, X_test,
                                                 y_train, y_validation, y_test):
    """
        The function should take in as arguments:
            - The model class
            - The training, validation, and test sets
            - A dictionary of hyperparameter names mapping to a list of values to be tried
        It should return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics.

    The dictionary of performance metrics should include a key called "validation_RMSE",
    for the RMSE on the validation set, which is what you should use to select the best model.
    Make sure that this function is general enough that it can be applied to other models.
    Note that the function should take in a model class, not an instance of that class, 
    so that it can initialise that class with the hyperparameters provided.
    """
    pass



