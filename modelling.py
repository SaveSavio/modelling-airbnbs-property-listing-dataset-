# TO DO: sort in alphabetical order
from tabular_data import load_airbnb
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model



import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
features, labels = load_airbnb(df, label="Price_Night")
print(features)
print(labels)


X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.3)

X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5)

print(X_train.shape, y_train.shape)

reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X_test, y_test)