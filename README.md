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
| SDGRegressor | 97.91 | 93.81 | 0.34 | 60.28 |
| DecisionTreeRegressor | 104.37 | 107.91 | 0.13 | 65.76 |
| RandomForestRegressor | 97.15 | 100.95 | 0.24 | 64.10 |
| GradientBoostingRegressor | 96.05 | 101.35 | 0.24 |  65.52 |

- The linear SGDRegressor is the baseline for our evaluation:
  - has a validation RMSE of 94$
  - R^2 is 0.34 meaning that only 34% of the Price/Night variance is explained by this model.

Best hyperparameters:

"alpha": 0.1, "eta0": 0.001, "learning_rate": "constant", "loss": "squared_error", "max_iter": 100000, "penalty": "elasticnet"

- The other 3 models performance is close. Anyway the SDG regressor shows the best performance whilst all the other models have a tendency to overfit meaning that they do not generalize.

This might be due to
- limitations in the data (e.g. noise)
- limitations in the hyperparameters tuning

It appears the regression models performance is limited by the amount of information provided by the numerical data.
The R^2 is always below 0.4 even for the best model, meaning our models have limited ability to explain the variability in the label.

### Regression models: outliers
![Boxplot](/price_night_boxplot.png)

The plot above showes the data are quite right skewed. Quite a few listings' nightly price is above the 3rd quartile. In order to improve the RMSE and yet use the numerical data only, we have split the data into a dataset without "Price_Night" outliers. The strategy consists in removing the data above the 90th-percentile (89 examples in total).

A dataset containing only the outliers is created as well. All the four models previously considered are fitted with the data with and without outliers.

Below, for simplicity, we only report the results for the SDG regressor (baseline) and the best performer (RandomForestRegressor).

| Estimator | Training RMSE | Validation RMSE | Test R^2 | Test MAE |
|----------|----------|----------|----------|----------|
| SDGRegressor w/o outliers| 49.61 | 44.34 | 0.29 | 35.33 |
| SDGRegressor outliers only | 173.66 | 184.83 | 0.1673 | 127.81 |
| RandomForestRegressor w/o outliers | 49.20 | 45.10 | 0.26 | 35.97 |
| RandomForestRegressor outliers only | 165.67 | 180.15 | 0.21 | 128.79 |

As expected, the RMSE is reduced when removing a portion of the outliers from the dataset as the model targets labels with lower variability.
It is debatable whether this approach could be useful or not in the real world when the label values is not known yet.

### Regression models: Neural Networks Regression

The file <u>modelling_NN_regression.py</u> implements a neural networks model to predict the "Price_Night" from numerical features (comprising one-hot-encoded).

The deep model performs better than the regressors tested before and generalizes better on unseen data (no overfitting).

| Estimator | Training RMSE | Validation RMSE | Validation R^2 | Validation MAE |
|----------|----------|----------|----------|----------|
| SDGRegressor | 97.91 | 93.81 | 0.34 | 60.28 |
| DecisionTreeRegressor | 104.37 | 107.91 | 0.13 | 65.76 |
| RandomForestRegressor | 97.15 | 100.95 | 0.24 | 64.10 |
| GradientBoostingRegressor | 96.05 | 101.35 | 0.24 |  65.52 |
| Neural Network | 93.14 | 85.82 | 0.43 |  65.35 |

The training_duration for the best model is 1.75s. Inference_latency equals to 5.76e-05s.

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
The chosen Key Performance Indicator is 'Accuracy'. Accuracy measures the proportion of correctly classified cases from the total number of objects in the dataset. To compute the metric, divide the number of correct predictions by the total number of predictions made by the model
(https://www.evidentlyai.com/classification-metrics/multi-class-metrics).

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \
$$

In the table below, we also report Precision, Recall, F1 score, all macro-averaged.

Results in the table below
| Estimator | Training Accuracy | Validation Accuracy | Validation Precision | Validation Recall | Validation F1
|----------|----------|----------|----------|----------|----------|
| LogisticRegression | 0.40 | 0.37 | 0.39 | 0.37 | 0.36 |
| GradientBoostingClassifier | 0.39 | 0.35 | 0.34 | 0.35 | 0.34 | 
| RandomForestClassifier | 0.40 | 0.36 | 0.36 | 0.35 | 0.34 | 
| DecisionTreeClassifier | 0.34 | 0.28 | 0.29 | 0.29 | 0.29 | 

The Logistic Regressor is our baseline model for simplicity and for performance. Yet is delivers, marginally, the best performance.
Logistic Regressor hyperparameters: {"C": 1.0, "max_iter": 1000, "penalty": "l2", "solver": "liblinear"}

All the models have a tendency to overfit. In that respect, the worst model is the DecisionTreeClassifier.

## Next steps
This framework is aimed to describe the workflow of Machine Learning. It uses an Occam Razor approach hence starting with simplest Linear Regression model then evolving to Neural Networks.

Main focus is on the training and tuning of models.

1) The performance of either regression or classification models is limited by the amount of information used.

4) A multimodal system that employs the information in the pictures as well as in the text description is expected to peform better than the current framework.