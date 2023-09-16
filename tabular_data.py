def clean_tabular_data():

    def remove_rows_with_missing_ratings(df):
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
        return df
    

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")
    df_clean = clean_tabular_data(df)
    df_clean.to_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")