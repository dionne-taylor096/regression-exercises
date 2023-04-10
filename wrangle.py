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
import wrangle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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
    filename = "final_zillow_single_family_properties_2017.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("SELECT propertylandusetypeid, parcelid, bedroomcnt, bathroomcnt,                     calculatedbathnbr, calculatedfinishedsquarefeet, fips, fullbathcnt,latitude, longitude,               lotsizesquarefeet, regionidzip, taxvaluedollarcnt, roomcnt,yearbuilt, assessmentyear,                 regionidcity, regionidcounty, propertylandusetype.propertylandusedesc,                                 predictions_2017.transactiondate FROM properties_2017 LEFT JOIN typeconstructiontype USING             (typeconstructiontypeid) LEFT JOIN propertylandusetype USING (propertylandusetypeid) LEFT JOIN         predictions_2017 USING(parcelid) LEFT JOIN architecturalstyletype                                     USING(architecturalstyletypeid) WHERE propertylandusedesc = 'Single Family Residential' OR             'Inferred Single Family Residential'",get_connection('zillow'))

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

#def split_data(df):
    #Split the data into training, testing, and validation sets
    #train, test = train_test_split(df, test_size=0.2, random_state=123)
    #train, validate = train_test_split(train, test_size=0.25, random_state=123)

    #return train, validate, test

def summarize_data(df):
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    #summary['column_name'] = summary.index
    summary = summary[['dtypes']]
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
                            'parcelid': 'parcel_id',
                            'bedroomcnt': 'num_bedrooms',
                            'bathroomcnt': 'num_bathrooms',
                            'calculatedfinishedsquarefeet': 'total_sqft',
                            'fips': 'county',
                            'regionidcity' : 'city',
                            'calculatedbathnbr' : 'ensuite_bathrooms',
                            'fullbathcnt': 'num_full_baths',
                            'latitude': 'lat',
                            'longitude': 'long',
                            'lotsizesquarefeet': 'lot_sqft',
                            'regionidzip': 'zip_code',
                            'taxvaluedollarcnt': 'assessed_property_value',
                            'roomcnt': 'num_rooms',
                            'yearbuilt': 'year_built',
                            'propertylandusedesc': 'home_type',
                            'transactiondate': 'last_date_sold'
})

    # Drop multiple columns at once
    df = df.drop(['Unnamed: 0','lat','long','zip_code','county','parcel_id'], axis=1)

    # Reset indexes
    #df = df.set_index('parcel_id')
    #df = df.reset_index('parcel_id')
    # reset the index
    df = df.reset_index()
    
    # Convert columns to appropriate data types
    df['assessed_property_value'] = df['assessed_property_value'].astype(float)
    df['total_sqft'] = df['total_sqft'].astype(int)
    df['year_built'] = df['year_built'].astype(int)
    df['lot_sqft'] = df['lot_sqft'].astype(int)
    df['ensuite_bathrooms'] = df['ensuite_bathrooms'].astype(float)
    df['num_rooms'] = df['num_rooms'].astype(float)
    df['num_full_baths'] = df['num_full_baths'].astype(float)

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

from sklearn.preprocessing import MinMaxScaler

def one_hot_encode_data(df):
    # One-hot encode categorical features
    categorical_features = ['home_type','county']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Reset indexes for both original and encoded DataFrames
    df = df.reset_index(drop=True)
    df_encoded = df_encoded.reset_index(drop=True)

    return df_encoded

def rename_encoded_columns(df):
    column_name_mapping = {
        'county_6059.0': 'Orange',
        'county_6111.0': 'Ventura',
        'city_4406.0': 'Los_Angeles',
        'city_5465.0': 'Anaheim',
        'city_5534.0': 'Brea',
        'city_118914.0': 'Long_Beach',
        'city_118994.0': 'Fullerton',
        'city_272578.0': 'Orange',
        'city_396053.0': 'Santa_Ana',
        'city_396054.0': 'Tustin',
        'city_396550.0': 'Ventura',
        'city_396551.0': 'West_Covina',
        'city_396556.0': 'Yorba_Linda'
    }
    df = df.rename(columns=column_name_mapping)
    return df


def scale_datas(X):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the features
    scaler.fit(X)

    # Transform the features using the fitted scaler
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.transform(X)

    return X_scaled

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
def split_datasets(X, y):
    from sklearn.model_selection import train_test_split

    # First, split the data into train (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Next, split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def split_dataset(df):
    from sklearn.model_selection import train_test_split
    
    # Separate the features (X) and target variable (y)
    X = df.drop(columns=['assessed_property_value_scaled'])
    y = df['assessed_property_value_scaled']

    # First, split the data into train (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Next, split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

# Call the function
#X_train, X_validate, X_test, y_train, y_validate, y_test = wrangle.split_dataset(df)

# Check the shape of X_train
#print(X_train.shape)


from sklearn.model_selection import train_test_split

def split_data(df):
   
    # Separate the features (X) and the target variable (y)
    X = df.drop('assessed_property_value', axis=1)
    y = df['assessed_property_value']

    # First, split the data into train (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Next, split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Concatenate the split data back into DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    validate_df = pd.concat([X_validate, y_validate], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Combine the train, validate, and test DataFrames into a single DataFrame
    df = pd.concat([train_df, validate_df, test_df], axis=0)

    return df


def drop_column(df, column_name):
    df = df.drop(column_name, axis=1)
    return df

def change_dtype(df, column_name, dtype):
    df[column_name] = df[column_name].astype(dtype)
    return df

def prepare_data(df):
    """
    Prepare the feature matrix X and the target variable y from the given DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the property data.
    
    Returns:
    X (pd.DataFrame): The feature matrix containing the selected features.
    y (pd.Series): The target variable, assessed property value.
    """
    X = df[['total_sqft', 'ensuite_bathrooms', 'num_rooms', 'year_built', 'num_full_baths', 'lot_sqft']]
    y = df['assessed_property_value']
    
    # Concatenate X and y into a single DataFrame
    df = pd.concat([X, y], axis=1)
    
    return df

def scaler_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the features (excluding the target variable)
    scaler.fit(df.drop(['assessed_property_value'], axis=1))

    # Transform all features except the target variable using the fitted scaler
    transformed_data = scaler.transform(df.drop(['assessed_property_value'], axis=1))

    print("Original DataFrame shape:", df.drop(['assessed_property_value'], axis=1).shape)
    print("Transformed DataFrame shape:", transformed_data.shape)

    # Set the transformed data to the original DataFrame
    df[df.drop(['assessed_property_value'], axis=1).columns] = transformed_data

    return df

def clean_up_X_y(df):
    y = df['assessed_property_value']
    df = df.drop('assessed_property_value', axis=1)
    return df

def prepared_data(df):
    """
    Prepare the feature matrix X and the target variable y from the given DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the property data.
    
    Returns:
    df (pd.DataFrame): The original DataFrame.
    X (pd.DataFrame): The feature matrix containing the selected features.
    y (pd.Series): The target variable, assessed property value.
    """
    X = df[['total_sqft', 'zip_code', 'county', 'home_type', 'city','num_rooms', 'year_built', 'num_full_baths', 'lot_sqft']]
    y = df['assessed_property_value']
    
    return df

def wrangle_zillow_data():
    df = get_zillow_data()
    df = drop_all_null(df)
    df = clean_zillow_data(df)
    df = prepare_data(df)
    #df = one_hot_encode_data(df)
    df = rename_encoded_columns(df)
    #df = scale_data(df)
    #df = scale_y_data(df)
    #df = clean_up_X_y(df)
    df = remove_outliers_iqr_loop(df, multiplier=1.5)
    df = split_data(df)
    summarize_data(df)
    #df = drop_missing_data(df, 0.9)
   
    return df

def wrangle_zillow_datas():
    df = get_zillow_data()
    df = clean_zillow_data(df)
    #X, y = prepare_data(df)
    #df = pd.concat([X, y], axis=1)
    df = scale_data(df)
    df = scale_y_data(df)
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


def regression_errors(y_true, y_pred):
    # Calculate the mean squared error
    mse = np.mean((y_pred - y_true) ** 2)

    # Calculate the root mean squared error
    rmse = np.sqrt(mse)

    # Calculate the explained variance score
    ssr = np.sum((y_pred - np.mean(y_true)) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    evs = 1 - (ssr / sst)

    # Calculate the R-squared score
    r2 = 1 - (np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    # Return the metrics as a dictionary
    return {'mse': mse, 'rmse': rmse, 'evs': evs, 'r2': r2}


def select_top_features(df, k=2):
    from sklearn.feature_selection import SelectKBest, f_regression
    
    # Select the top k features for predicting tip amount
    selector = SelectKBest(f_regression, k=k)
    X = df[['total_sqft','zip_code','num_rooms','year_built','num_full_baths','lot_sqft']]
    y = df['assessed_property_value']
    X_kbest = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    # Print the top k features
    print('Top', k, 'Features:', list(selected_features))

    # Return the top k features as a list
    return list(selected_features)

def cv_evaluation(df):
    # Prepare your data
    X = df.drop('assessed_property_value', axis=1)
    y = df['assessed_property_value']

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale your features if necessary
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a list of models to test
    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor()),
        ('Gradient Boosting', GradientBoostingRegressor()),
        ('XGBoost', XGBRegressor())
    ]

    # Loop through the models and evaluate each one using cross-validation
    for name, model in models:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5,                                         scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        print(f"{name}: Avg. RMSE: {np.mean(rmse_scores):.4f}, Std. Dev: {np.std(rmse_scores):.4f}")


def check_feature_importance(df):
    X = df.drop('assessed_property_value', axis=1)
    y = df['assessed_property_value']

    model = LinearRegression()
    model.fit(X, y)

    # Pair the column names with their respective coefficients
    feature_importances = list(zip(X.columns, model.coef_))

    # Print the feature importances
    for feature, importance in feature_importances:
        print(f"{feature}: {importance}")

    #return feature_importances

def feature_elimination(df, n_features_to_select=3):
    
    X = df.drop('assessed_property_value', axis=1)
    y = df['assessed_property_value']
    
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    # Get the column names for the features that RFE has selected
    selected_columns = X.columns[rfe.support_]

    # Print the selected columns
    print("Selected columns:", selected_columns)
    
    #return selected_columns
    
    
def hist_frequency_chart(df, continuous_vars=['total_sqft', 'year_built', 'lot_sqft', 'num_rooms']):
    for var in continuous_vars:
        plt.figure()
        plt.hist(df[var], bins=20)
        plt.title(f'Histogram of {var}')
        plt.xlabel(var)
        plt.ylabel('Frequency')
        plt.show()
        
def get_stats(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X = sm.add_constant(X)  # Add a constant term for the intercept
    OLS = sm.OLS(y, X).fit()
    print(OLS.summary())
    return OLS

def model_selection(df, target_col):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor()),
        ('Gradient Boosting', GradientBoostingRegressor()),
        ('XGBoost', XGBRegressor())
    ]

    for name, model in models:
        mse = -np.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
        r2 = np.mean(cross_val_score(model, X, y, scoring='r2', cv=5))
        print(f"{name}: Mean Squared Error = {mse:.4f}, R^2 Score = {r2:.4f}")
        
        
def model_eval(df, target_column):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    feature_columns = [col for col in df.columns if col != target_column]

    X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size=0.2, random_state=42)

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor())
    ]

    def get_p_values(X, y, model):
        n = X.shape[0]
        p = X.shape[1]
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        MSE = RSS / (n - p - 1)

        se_beta = np.sqrt(np.diagonal(MSE * np.linalg.inv(np.dot(X.T, X))))
        t_stat = model.coef_ / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p - 1))

        return p_values

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

    for feature in feature_columns:
        sns.regplot(x=feature, y=target_column, data=df)
        plt.title(f"{feature} vs. {target_column}")
        plt.show()

    p_values = get_p_values(X_train, y_train, best_model[1])
    for feature, p_value in zip(feature_columns, p_values):
        print(f"P>|t| for {feature}: {p_value:.3f}")

        
def model_interpretation(df, target_column, best_params):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    
    best_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True
    }
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(**best_params)

    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})

    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    print(feature_importances)


def feature_importance_bar_chart(feature_importances):

    plt.figure(figsize=(10, 5))
    plt.bar(feature_importances['feature'], feature_importances['importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=45)
    plt.show()
    
def model_validation(df, target_column):
    feature_columns = [col for col in df.columns if col != target_column]

    X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size=0.2, random_state=42)

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor())
    ]

    def get_p_values(X, y, model):
        n = X.shape[0]
        p = X.shape[1]
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        MSE = RSS / (n - p - 1)

        se_beta = np.sqrt(np.diagonal(MSE * np.linalg.inv(np.dot(X.T, X))))
        t_stat = model.coef_ / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p - 1))

        return p_values

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

    p_values = get_p_values(X_train, y_train, best_model[1])
    for feature, p_value in zip(feature_columns, p_values):
        print(f"P>|t| for {feature}: {p_value:.3f}")
        
        
def feature_correlation(df):
    # Remove 'assessed_property_value' column from the list of features
    features = [column for column in df.columns if column != 'assessed_property_value']

    # Set the significance level
    alpha = 0.05

    # Loop through all features and perform hypothesis test
    for feature in features:
        X = df[feature]
        y = df['assessed_property_value']

        # Calculate Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = stats.pearsonr(X, y)
        correlation_coefficient_s, p_value_s = stats.spearmanr(X, y)

        # Print the results
        print(f"Feature: {feature}")
        print(f"Pearson correlation coefficient: {correlation_coefficient}")
        print(f"P-value: {p_value}")
        
        print("\n")
        print(f"Spearman correlation coefficient: {correlation_coefficient_s}")
        print(f"P-value: {p_value_s}")

        # Test the null hypothesis
        if p_value < alpha:
            print("\033[32mReject the null hypothesis (H0): There is a linear relationship'.\033[0m")
        else:
            print("\033[31mFail to reject the null hypothesis (H0): There is no evidence of a linear relationship'.\033[0m")
        
        print("\n")
        
        
def data_dict():
    data_dict = {
        'id': 'Unique identifier for each property',
        'parcelid': 'Unique identifier for each property, used in conjunction with "assessmentyear" to form a composite primary key',
        'bedroomcnt': 'Number of bedrooms in the property',
        'bathroomcnt': 'Number of bathrooms in the property',
        'fireplacecnt': 'Number of fireplaces in the property',
        'calculatedbathnbr': 'Number of bathrooms in the property (including fractional bathrooms)',
        'calculatedfinishedsquarefeet': 'Total finished living area of the property, in square feet',
        'fullbathcnt': 'Number of full bathrooms in the property (including fractional bathrooms)',
        'garagecarcnt': 'Number of cars that can fit in the garage, if applicable',
        'garagetotalsqft': 'Total square footage of the garage, if applicable',
        'latitude': 'Latitude of the property',
        'longitude': 'Longitude of the property',
        'lotsizesquarefeet': 'Total area of the lot, in square feet',
        'regionidzip': 'Zip code of the property',
        'taxvaluedollarcnt': 'Total tax assessed value of the property, in dollars',
        'roomcnt': 'Total number of rooms in the property (including bedrooms and bathrooms)',
        'yearbuilt': 'Year the property was built',
        'numberofstories': 'Number of stories in the property, if applicable',
        'assessmentyear': 'Year of the property assessment, used in conjunction with "parcelid" to form a composite primary key',
        'landtaxvaluedollarcnt': 'Tax assessed value of the land, in dollars',
        'structuretaxvaluedollarcnt': 'Tax assessed value of the structure, in dollars',
        'taxamount': 'Total property tax for the assessment year, in dollars'
    }

    data_dict_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Description'])
    return data_dict_df