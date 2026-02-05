# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %%
# Shared Helper Functions:

# %%
# Drop features with high amounts of null values
def drop_high_missing(df, threshold=90):
    missing_pct = df.isnull().mean() * 100
    return df.drop(columns=missing_pct[missing_pct > threshold].index)

# %%
# Normalize numeric features with min-max scaler
def scale_numeric(df):
    numeric_cols = df.select_dtypes('number').columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# %%
# Hot-one encode categorical features
def encode_categoricals(df):
    cat_cols = df.select_dtypes('category').columns
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

# %%
# Dataset Specific Functions:

# %%
# Prepare Graduation Dataset
def prepare_graduation_features(df, cols_to_drop):
    # Remove high-missing columns
    df = drop_high_missing(df)

    # Remove rows missing target
    df = df.dropna(subset=['grad_150_value'])

    # Separate target
    y = df['grad_150_value']
    X = df.drop(columns=['grad_150_value'])

    # Convert categoricals
    categorical_cols = [
        'chronname', 'city', 'state',
        'level', 'control', 'basic', 'similar'
    ]
    X[categorical_cols] = X[categorical_cols].astype('category')

    # Drop unneeded columns
    X = X.drop(columns=cols_to_drop)

    # Scale and encode
    X = scale_numeric(X)
    X = encode_categoricals(X)

    return X, y

# %%
# Split Graduation Data
def split_regression_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test

# %%
# Prepare Placement Data
def prepare_placement_features(df):
    # Drop salary
    df = df.drop(columns=['salary'])

    # Create target
    df['placed'] = (df.status == 'Placed').astype(int)
    y = df['placed']

    # Convert categoricals
    categorical_cols = [
        'gender', 'ssc_b', 'hsc_b',
        'hsc_s', 'degree_t', 'workex',
        'specialisation'
    ]
    df[categorical_cols] = df[categorical_cols].astype('category')

    # Drop non-features
    X = df.drop(columns=['sl_no', 'status', 'placed'])

    # Scale + encode
    X = scale_numeric(X)
    X = encode_categoricals(X)

    return X, y

# %%
# Split Placement Data
def split_classification_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.70, stratify=y, random_state=42
    )

    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, train_size=0.50, stratify=y_temp, random_state=42
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test