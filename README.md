# Modelling Airbnb's property listing dataset 

<u>Motivation</u>:<br>
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

## Installation
Ensure the following packates are installed or run the following commands using pip package installer (https://pypi.org/project/pip/):

```python
pip install pandas
pip install numpy
pip install scikit-learn
pip install joblib
```
The remaining packages are part of Python's standard library, so no additional installation is needed.

## Data Preparation
The data comes in form of:
- tabular data ('AirBnbData.csv' file)
- images (a series of '.png' files, one folder for each listing)

The tabular dataset has the following columns:

- ID: Unique identifier for the listing
- Category: The category of the listing
- Title: The title of the listing
- Description: The description of the listing
- Amenities: The available amenities of the listing
- Location: The location of the listing
guests: The number of guests that can be accommodated in the listing
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


In order to prepare the tabular data, a function
```python
def clean_tabular_data(df)
```
that takes a pandas dataframe containing the AirBnb listings and cleanes it by calling three separate functions:
```python
def remove_rows_with_missing_ratings(df)
def combine_description_strings(df)
def set_default_feature_values(df)
```
which, in turn:
- Remove the rows with missing values in each of the rating columns
- Combines the list items into the same string
- Replaces the empty rows in the colums "guests", "beds", "bathrooms", "bedrooms" with a default value equal to 1

These functions are placed inside <u>tabular_data.py</u>:<br> that, when called, performs the cleaning if called and saves the clean data in clean_tabular_data.csv.

The function
```python
def load_airbnb(df, label="label", numeric_only=False):
```
splits the dataframe into features and labels in order to prepare it for Machine Learning.
Furthermore, if required, can remove all non-numeric values from the dataset.