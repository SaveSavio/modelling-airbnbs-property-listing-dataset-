import pandas as pd

class database_utils():
    """
        Cleans the Airbnb dataset before analysis.
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
        
        df = df[df["Cleanliness_rating"]<=5] # drop a column with wrong rating
        return df

    def combine_description_strings(df):
        """
            Combines the list items in the description column into the same string,
            after applying some cleaning to remove redundant information.
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

    def category_one_hot_encoding(df):
        """
            Encodes the column 'Category'
            Parameters:
                a pandas dataframe
            Returns:
                a pandas dataframe with additional one-hot-encoding columns
        """
        df['Category'] = df['Category'].astype('category')
        df = pd.get_dummies(df, columns=['Category'], dtype=int, prefix=['Category'])
        return df

    def location_one_hot_encoding(df):
        """
            Encodes the column 'Location'.
            At first, it extracts the nation from the location column.
            Then maps each nation to a geographical area (continent, in this case)
            Parameters:
                a pandas dataframe
            Returns:
                a pandas dataframe
        """
        df["Location"] = df['Location'].str.replace('Czech Republic', 'Czech_Republic')
        df["Location"] = df['Location'].str.replace('Dominican Republic', 'Dominican_Republic')
        
        country_to_area = {
            'Kingdom': 'Europe',
            'France': 'Europe',
            'Netherlands': 'Europe',
            'States': 'North America',
            'Germany': 'Europe',
            'Spain': 'Europe',
            'Norway': 'Europe',
            'Romania': 'Europe',
            'Latvia': 'Europe',
            'Lithuania': 'Europe',
            'Croatia': 'Europe',
            'Belgium': 'Europe',
            'Sweden': 'Europe',
            'Estonia': 'Europe',
            'Guadeloupe': 'Central America',
            'Rica': 'Central America',
            'Colombia': 'South America',
            'Poland': 'Europe',
            'Canada': 'North America',
            'Portugal': 'Europe',
            'Austria': 'Europe',
            'Czech_Republic': 'Europe',
            'Greece': 'Europe',
            'Panama': 'Central America',
            'Rico': 'Central America',
            'Lucia': 'Central America',
            'Italy': 'Europe',
            'Dominican_Republic': 'Central America',
            'Nicaragua': 'Central America',
            'Tobago': 'Central America',
            'Jersey': 'Europe',
            'Peru': 'South America',
            'Korea': 'Asia',
            'Finland': 'Europe',
            'Belize': 'Central America',
            'Australia': 'Australia',
            'Indonesia': 'Asia',
            'Thailand': 'Asia',
            'Mexico': 'Central America',
            'Zealand': 'Australia',
            'Chile': 'South America',
            'Malaysia': 'Asia',
            'Turkey': 'Asia',
            'India': 'Asia',
            'South Africa': 'Africa',
            'Philippines': 'Asia',
            'Brazil': 'South America',
            'Ukraine': 'Europe',
            'Ireland': 'Europe',
            'Ecuador': 'South America',
            'Luxembourg': 'Europe',
            'Japan': 'Asia',
            'China': 'Asia',
            'Africa': 'Africa',
            'Czechia': 'Europe'
        }        
        df['Geographical_Area'] = df['Location'].str.split().str[-1].map(country_to_area)
        return df

    def reduce_skewness(df):
        """
            Performs log trasform on selected data: Price_Night, 
            Parameters:
                a pandas dataframe
            Returns:
                a pandas dataframe
        """
        df["Price_Night_Log"] = df["Price_Night"].map(lambda i: np.log(i) if i > 0 else 0)
        df["guests_Log"] = df["guests"].map(lambda i: np.log(i) if i > 0 else 0)
        df = df.drop(columns=["Price_Night", "guests"])
        return df


    def load_airbnb(df, label="label", numeric_only=False):
        """
            this function performs two tasks:
            1) Selects numerical data only and returns two dataframes
            2) Splits features and labels
            Parameters:
                A pandas dataframe; the name of the label
            Returns:
                A tuple containing model features (numeric only)
                A tuple containing the model label
        """
        labels = df[label]
        features = df.drop(columns=[label, 'Unnamed: 0', 'Unnamed: 19'])
        # remove all non-numerical features from the features
        if numeric_only == True:
            features = features.select_dtypes(include='number')
        return (features, labels)


if __name__ == "__main__":
    df = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")
    df_clean = database_utils.remove_rows_with_missing_ratings(df)
    df_clean = database_utils.combine_description_strings(df_clean)
    df_clean = database_utils.set_default_feature_values(df_clean)
    df_clean.to_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    
    df_clean_transformed = database_utils.category_one_hot_encoding(df_clean)
    df_clean_transformed = database_utils.location_one_hot_encoding(df_clean_transformed)
    df_clean_transformed.to_csv("./airbnb-property-listings/tabular_data/clean_tabular_data_transformed.csv")
