from sklearn import linear_model
from tabular_data import load_airbnb
from sklearn import model_selection
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import StandardScaler # need to understand if best using this
from sklearn.linear_model import SGDRegressor

df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
features, labels = load_airbnb(df, label="Price_Night")

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit and transform the data
scaled_features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    scaled_features, labels, test_size=0.3)

X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5)

print(f'The number of training examples is {X_train.shape[0]}\n The number of labels is {X_train.shape[1]}')

# create a simple linear regression model
# model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))
model = SGDRegressor(max_iter=10000, tol=1e-3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred)
print('RMSE = ', np.sqrt(mse))

r2 = metrics.r2_score(y_test, y_pred)
print('rˆ2 = ', r2)

import itertools
import typing

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


grid = {
    "n_estimators": [10, 50, 100],
    "criterion": ["mse", "mae"],
    "min_samples_leaf": [2, 1],
}

for i, hyperparams in enumerate(grid_search(grid)):
    print(i, hyperparams)

def custom_tune_regression_model_hyperparameters(SGDregressor):
    pass

