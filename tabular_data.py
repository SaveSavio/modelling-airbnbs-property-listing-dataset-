def clean_tabular_data(df):
    """
        Main Function to clean the Airbnb dataset before analysis
    """
    def remove_rows_with_missing_ratings(df):
        """
            Removes the rows with missing values in each of the rating columns.
            Parameters:
                a pandas dataframe
            Returns:
                the same type
        """
        df = df[df['Cleanliness_rating'].notna()]
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
    
    def load_airbnb(df, label="label"):
        df = df.select_dtypes(include='number')
        labels = df[label]
        features = df.drop(df.drop(columns=[label]))
        return (features, labels)

    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)

    return df

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")
    df_clean = clean_tabular_data(df)
    df_clean.to_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
