#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import env
from env import host, user, password
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# In[ ]:


def get_connection(db):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[ ]:


def get_data():
    # Check if the data has already been cached
    cache_file = input("Enter a name for the CSV file: ")
    cache_file_csv = cache_file + ".csv"
    user = env.user
    password = env.password
    host = env.host
    #db = input('Enter the name of the database you want to access: ')
    #table = input('Enter the name of the table you want to access: ')
    
    if os.path.isfile(cache_file_csv):
        print(f'Loading data from {cache_file_csv}')
        df = pd.read_csv(cache_file_csv)
        print(df)
    else: 
        print("File doesn't exist.")
        db = input('Enter the name of the database you want to access: ')
        print("Establishing connection and diplaying query")
        conn = get_connection(db)
        table = input('Enter the name of the table you want to access: ')
        print("Diplaying query")
        
        # query and open table in pandas
        df = pd.read_sql(f'SELECT * FROM {table}', conn)
        
        # Cache the data by writing it to a CSV file
        new_cache_file = input("Enter a name for the CSV file to cache the data: ")
        new_cache_file_csv = new_cache_file + ".csv"
        df.to_csv(new_cache_file_csv, index=False)
        print(f'Saved data to {new_cache_file_csv}')
    return df


# In[ ]:
    
def get_zillow_data():
    filename = "zillow_single_family_props_2017.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("SELECT propertylandusetypeid, id, parcelid, bedroomcnt, bathroomcnt, fireplacecnt, calculatedbathnbr, calculatedfinishedsquarefeet, fullbathcnt, garagecarcnt, garagetotalsqft, latitude, longitude, lotsizesquarefeet, regionidzip, taxvaluedollarcnt, roomcnt, yearbuilt, numberofstories, assessmentyear, landtaxvaluedollarcnt, structuretaxvaluedollarcnt, taxamount, propertylandusetype.propertylandusedesc FROM properties_2017 LEFT JOIN typeconstructiontype USING (typeconstructiontypeid) LEFT JOIN propertylandusetype USING (propertylandusetypeid) WHERE propertylandusedesc = 'Single Family Residential'",get_connection('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        
        # Return the dataframe to the calling code
        return df 
    


# In[ ]:

def convert_to_integer(df, columns):
    for col in columns:
        df[col] = df[col].fillna(-1).astype(int)
        df.loc[df[col] == -1, col] = np.nan
    return df

def standardize_text(df, columns):
    for col in columns:
        df[col] = df[col].str.upper().str.strip()
    return df

def detect_column_types(df):
    """
    Returns a dictionary with column names grouped by their data types.
    """
    column_info = df.dtypes.groupby(df.dtypes).groups
    column_groups = {}
    for dtype, column_list in column_info.items():
        column_groups[dtype] = column_list.tolist()
    return column_groups



def encode_categorical_columns(df, categorical_columns, encoding_method='ordinal'):
    """
    Encodes categorical columns using the specified encoding method.
    """
    if encoding_method == 'ordinal':
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    # Add other encoding methods if needed
    return df


def change_numerical_columns_datatype(df, numerical_columns, datatype='float64'):
    """
    Changes the datatype of numerical columns.
    """
    for column in numerical_columns:
        df[column] = df[column].astype(datatype)
    return df


def encode_binary_columns(df, columns, encoding_method='ordinal'):
    """
    Encodes binary columns using the specified encoding method.
    """
    if encoding_method == 'ordinal':
        for col in columns:
            unique_values = df[col].unique()
            value_map = {value: i for i, value in enumerate(unique_values)}
            df[col] = df[col].replace(value_map).astype(int)
    # Add other encoding methods as needed
    return df

def get_numerical_columns(df):
    """
    Returns a list of column names for numerical columns.
    """
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
    return numerical_columns

def get_categorical_columns(df):
    """
    Returns a list of column names containing categorical data in the given DataFrame.
    """
    object_columns = df.select_dtypes(include=['object']).columns.to_list()
    boolean_columns = df.select_dtypes(include=['bool']).columns.to_list()
    category_columns = df.select_dtypes(include=['category']).columns.to_list()
    categorical_columns = object_columns + boolean_columns + category_columns
    return categorical_columns

def get_features(df):
    """
    Returns the first n column names of the DataFrame.
    """
    return df.columns[:n]


# In[ ]:

def split_data(df):
    #Split the data into training, testing, and validation sets
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train, test_size=0.25, random_state=123)

    return train, validate, test

def summarize_data(df):
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['column_name'] = summary.index
    summary = summary[['column_name', 'dtypes']]
    summary['missing_values'] = df.isnull().sum().values
    summary['unique_values'] = df.nunique().values
    summary['count'] = df.count().values
    summary['mean'] = df.mean().values
    summary['std'] = df.std().values
    summary['min'] = df.min().values
    summary['25%'] = df.quantile(0.25).values
    summary['50%'] = df.quantile(0.5).values
    summary['75%'] = df.quantile(0.75).values
    summary['max'] = df.max().values
    return summary

def clean_zillow_data(df):
   
    # Rename columns
    df = df.rename(columns={'propertylandusetypeid': 'property_type_id',
                            'id': 'record_id',
                            'bedroomcnt': 'num_bedrooms',
                            'bathroomcnt': 'num_bathrooms',
                            'fireplacecnt': 'num_fireplaces',
                            'calculatedfinishedsquarefeet': 'total_sqft',
                            'fullbathcnt': 'num_full_baths',
                            'garagecarcnt': 'num_garage_cars',
                            'garagetotalsqft': 'garage_sqft',
                            'latitude': 'lat',
                            'longitude': 'long',
                            'lotsizesquarefeet': 'lot_sqft',
                            'regionidzip': 'zip_code',
                            'taxvaluedollarcnt': 'assessed_property_value',
                            'roomcnt': 'num_rooms',
                            'yearbuilt': 'year_built',
                            'numberofstories': 'num_stories',
                            'assessmentyear': 'assessment_year',
                            'landtaxvaluedollarcnt': 'land_tax_value',
                            'structuretaxvaluedollarcnt': 'structure_tax_value',
                            'taxamount': 'tax_amount'})

    # Drop multiple columns at once
    df = df.drop(['Unnamed: 0','calculatedbathnbr', 'num_garage_cars', 'num_stories', 
                  'num_fireplaces', 'garage_sqft', 'propertylandusedesc', 'parcelid',                                   'structure_tax_value', 'land_tax_value', 'tax_amount', 'lat','long',                                   'property_type_id', 'assessment_year','num_bathrooms','num_bedrooms' ], axis=1)

    # Reset indexes
    df = df.set_index('record_id')
    df = df.reset_index('record_id')
    
    # Convert columns to appropriate data types
    df['assessed_property_value'] = df['assessed_property_value'].astype(float)
    df['total_sqft'] = df['total_sqft'].astype(int)
    df['year_built'] = df['year_built'].astype(int)
    df['lot_sqft'] = df['lot_sqft'].astype(int)

    return df

def drop_missing_data(df, threshold):
    
    #Drops rows from the dataframe where the percentage of missing values is above the thre
    #Args:
    #df (pd.DataFrame): the dataframe to process.
    #threshold (float): the percentage threshold above which rows will be dr
    #Returns:
    #pd.DataFrame: the processed dataframe.
    
    # Calculate the number of non-missing values in each row.
    num_non_missing = df.count(axis=1)
    
    # Calculate the percentage of missing values in each row.
    pct_missing = 1 - num_non_missing / len(df.columns)
 
    # Drop rows where the percentage of missing values is above the threshold.
    df = df.loc[pct_missing <= threshold]
 
    #df = drop_missing_data(df, 0.5)
    return df

def drop_all_null(df):
    df = df.dropna()
    df = df.dropna(how='all')
    return df


def remove_outliers_iqr(df, multiplier=1.5):
    # Select only numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Loop through numeric columns
    for col in num_cols:
        # Calculate the IQR of the column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the upper and lower bounds for outliers
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Filter out the outliers from the DataFrame
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def remove_outliers_iqr_loop(df, multiplier=1.5):
    # Select only numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Initialize the number of outliers
    num_outliers = 1
    
    while num_outliers > 0:
        # Initialize the number of outliers
        num_outliers = 0
        
        # Loop through numeric columns
        for col in num_cols:
            # Calculate the IQR of the column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define the upper and lower bounds for outliers
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Filter out the outliers from the DataFrame
            num_outliers += df[col][(df[col] < lower_bound) | (df[col] > upper_bound)].count()
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def standardize_text(df, columns):
    for col in columns:
        df[col] = df[col].str.upper().str.strip()
    return df


def scale_x_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the 'county' column
    scaler.fit(df[['total_sqft']])

    # Transform the 'county' column using the fitted scaler
    df['total_sqft_scaled'] = scaler.transform(df[['total_sqft']])
  
    return df

def scale_y_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the 'county' column
    scaler.fit(df[['assessed_property_value']])

    # Transform the 'county' column using the fitted scaler
    df['assessed_property_value_scaled'] = scaler.transform(df[['assessed_property_value']])
    return df

def scale_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to all features except the target variable
    scaler.fit(df.drop(['assessed_property_value'], axis=1))

    # Transform all features except the target variable using the fitted scaler
    df[df.drop(['assessed_property_value'], axis=1).columns] = scaler.transform(df.drop(['assessed_property_value'], axis=1))

    return df

def define_X_y(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y

#X, y = define_X_y(df, 'assessed_property_value')
#X = X[['total_sqft', 'lot_sqft', 'num_rooms', 'year_built']]


#def split_data(df):
    
    # First, split the data into train (80%) and a temporary set (20%)
    #X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Next, split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    #X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def drop_column(df, column_name):
    df = df.drop(column_name, axis=1)
    return df

def change_dtype(df, column_name, dtype):
    df[column_name] = df[column_name].astype(dtype)
    return df

def wrangle_zillow_data():
    df = get_zillow_data()
    df = drop_all_null(df)
    df = clean_zillow_data(df)
    #df = drop_missing_data(df, 0.9)
   
    return df

def make_sample(df, frac=0.003):
    """
    This function takes a DataFrame as input and returns a random sample of the data.
    
    Parameters
    ----------
    df : DataFrame
        The input DataFrame to take a random sample from.
    frac : float, optional
        The fraction of the data to take as a sample. Default is 0.001 (0.1%).
        
    Returns
    -------
    DataFrame
        The random sample of the input DataFrame.
    """
    sample_df = df.sample(frac=frac)
    return sample_df

# Example usage
# sample_df = make_sample(df, frac=0.001)


def count_outliers_iqr(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((df < lower_bound) | (df > upper_bound)).sum().sum()
    return outliers





