import pandas as pd

def clean_tabular_data(df):
    """
        Main Function to clean the Airbnb dataset before analysis.
        It calls the nested functions on a pandas dataframes
        Parameters:
            a pandas dataframe
        Returns:
            a "clean" pandas dataframe
    """
    def remove_rows_with_missing_ratings(df):
        """
            Removes the rows with missing values in each of the rating columns.
            Parameters:
                a pandas dataframe
            Returns:
                a pandas dataframe
        """
        df = df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating',
                               'Check-in_rating', 'Value_rating'], axis=0, how='any')
        return df

    def combine_description_strings(df):
        """
            Combines the list items into the same string
            Parameters:
                a pandas dataframe
            Returns:
                a pandas dataframe
        """
        df['Description'] = df['Description'].str.replace("'About this space', ", '')
        df['Description'] = df['Description'].str.replace(" 'The space', 'The space\n", '')
        df['Description'] = df['Description'].str.replace(r'\n\n', ' ')
        df['Description'] = df['Description'].str.replace(r'\n', ' ')
        df['Description'] = df['Description'].replace("''", "")
        return df

    def set_default_feature_values(df):
        """
        Replaces the empty rows in the colums "guests", "beds", "bathrooms", "bedrooms"
        with a default value equal to 1
            Parameters:
                a pandas dataframe
            Returns:
                a pandas dataframe
        """
        df[['guests', 'beds', 'bathrooms', 'bedrooms']] = df[['guests', 'beds', 'bathrooms', 'bedrooms']].fillna(value=1)
        return df

    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)

    return df


def load_airbnb(df, label="label", numeric_only=False):
    """
        1) Selects numerical data only and returns two dataframes
        2) Splits features and labels.
        Parameters:
            A pandas dataframe; the name of the label
        Returns:
            A tuple containing model features (numeric only)
            A tuple containing the model label
    """
    # remove all non-numerical features from the dataset
    if numeric_only == True:
        df = df.select_dtypes(include='number')
    labels = df[label]
    features = df.drop(columns=[label, 'Unnamed: 0', 'Unnamed: 19'])
    return (features, labels)

if __name__ == "__main__":
    df = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")
    df_clean = clean_tabular_data(df)
    df_clean.to_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
