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


def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_file(filename)

        # Return the dataframe to the calling code
        return df  


# In[ ]:


def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM species JOIN measurements USING (species_id)', get_connection('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  


# In[ ]:


def get_telco_data():
    filename = "telco_churn.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM customers LEFT JOIN contract_types USING (contract_type_id)LEFT JOIN internet_service_types USING (internet_service_type_id) LEFT JOIN payment_types USING (payment_type_id)', get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  
    
def get_zillow_data():
    filename = "single_family_properties_2017.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017 LEFT JOIN propertylandusetype USING (propertylandusetypeid) WHERE propertylandusetypeid = '261' OR '279'", get_connection('zillow'))

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
    categorical_columns = object_columns + boolean_columns
    return categorical_columns

def get_features(df):
    """
    Returns the first n column names of the DataFrame.
    """
    return df.columns[:n]


# In[ ]:

def split_data(df):
    # Split the data into training, testing, and validation sets
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train, test_size=0.25, random_state=123)

    return train, validate, test

def summarize_data(df):
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['column_name'] = summary['index']
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
   # Fill missing values in the 'bathroomcnt' and 'bedroomcnt' columns with 0
    df.loc[:, ['bathroomcnt', 'bedroomcnt']] = df[['bathroomcnt', 'bedroomcnt']].fillna(0)

    # Fill missing values in the 'taxvaluedollarcnt' and 'taxamount' columns with 0
    df.loc[:, ['taxvaluedollarcnt', 'taxamount']] = df[['taxvaluedollarcnt', 'taxamount']].fillna(0)
    
    # Remove rows with more than a certain percentage of NaN values, e.g., 50%
    threshold = 0.5
    df = df.dropna(thresh=int(threshold * len(df.columns)))
    
    # Drop rows where more than half of the columns are missing
    df = df.dropna(thresh=int(df.shape[1] / 2))
    # Rename columns
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'total_sqft', 'taxvaluedollarcnt': 'assessed_property_value', 'bathroomcnt': 'num_bathrooms', 'bedroomcnt': 'num_bedrooms', 'fips':'county', 'yearbuilt':'year_built', 'taxamount':'total_property_tax'})

    df['assessed_property_value'] = df['assessed_property_value'].astype(float)
    
    # Convert the column to integer data type using .astype()
    # Convert the column to integer data type using .astype()
    #df['total_sqft'] = df['total_sqft'].astype(int)
  
    
    df = df.dropna(subset=['total_sqft'])
    df['total_sqft'] = df['total_sqft'].astype(int)
    
    df = df.dropna(subset=['county'])
    df['county'] = df['county'].astype(int)
    
    df = df.dropna(subset=['year_built'])
    df['year_built'] = df['year_built'].astype(int)
  
    # Drop a single column
    df = df.drop('Unnamed: 0', axis=1)
    
    # Create the OrdinalEncoder object
    #encoder = OneHotEncoder()

    # Fit and transform the 'category' column
    #encoded_column = encoder.fit_transform(df[['county']])

    # Replace the original 'category' column with the encoded one
    #df['county'] = encoded_column
   
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

def standardize_text(df, columns):
    for col in columns:
        df[col] = df[col].str.upper().str.strip()
    return df


def scale_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the 'county' column
    scaler.fit(df[['county']])

    # Transform the 'county' column using the fitted scaler
    df['county_scaled'] = scaler.transform(df[['county']])
    return df

def split_data(X, y):
    # First, split the data into train (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Next, split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def drop_column(df, column_name):
    df = df.drop(column_name, axis=1)
    return df

def change_dtype(df, column_name, dtype):
    df[column_name] = df[column_name].astype(dtype)
    return df

def wrangle_zillow_data():
    df = get_zillow_data()
    df = clean_zillow_data(df)
    return df

def compare_scalers(df, features, target, test_size=0.2, random_state=42):
    
    # Define the target variable and features
    target = 'total_property_tax'
    features = [col for col in df.columns if col != target]
    
    X = df[features]
    y = df[target]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the scalers to test
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }

    # Initialize a linear regression model
    lm = LinearRegression()

    results = []

    # Iterate over each scaler, apply it to the data, and evaluate the model
    for scaler_name, scaler in scalers.items():
        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit the model on the scaled data
        lm.fit(X_train_scaled, y_train)

        # Calculate the test mean squared error
        y_pred = lm.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Store the results
        results.append({
            'Scaler': scaler_name,
            'MSE': mse,
            'RMSE': rmse
        })

    # Create a DataFrame with the results and return it
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Scaler')

    return results_df





