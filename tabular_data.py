def clean_tabular_data(df):
    """
        Main Function to clean the Airbnb dataset before analysis
    """
    # TO DO: clarify if the 3 functions below should be indented or not
    def remove_rows_with_missing_ratings(df):
        """
            Removes the rows with missing values in each of the rating columns.
            Parameters:
                a pandas dataframe
            Returns:
                the same type
        """
        # TO DO: remove line below once function is tested
        # df = df[df['Cleanliness_rating'].notna()]
        df = df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating',
                               'Check-in_rating', 'Value_rating'], axis=0, how='any')
        return df

    def combine_description_strings(df):
        df['Description'] = df['Description'].str.replace("'About this space', ", '')
        df['Description'] = df['Description'].str.replace(" 'The space', 'The space\n", '')
        df['Description'] = df['Description'].str.replace(r'\n\n', ' ')
        df['Description'] = df['Description'].str.replace(r'\n', ' ')
        df['Description'] = df['Description'].replace("''", "")
        return df

    def set_default_feature_values(df):
        df[['guests', 'beds', 'bathrooms', 'bedrooms']] = df[['guests', 'beds', 'bathrooms', 'bedrooms']].fillna(value=1)
        return df

    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)

    return df

def load_airbnb(df, label="label", numeric_only=False):
    """
        Selects numberical data only and returns two dataframes, splitting features from labels.
        Parameters:
            A pandas dataframe; the name of the label
        Returns:
            A tuple containing model features (numeric only)
            A tuple containing the model labels
    """
    # remove all non-numerical features from the dataset
    if numeric_only == True:
        df = df.select_dtypes(include='number')
    labels = df[label]
    features = df.drop(columns=[label, 'Unnamed: 0', 'Unnamed: 19'])
    return (features, labels)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")
    df_clean = clean_tabular_data(df)
    df_clean.to_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
