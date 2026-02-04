# Max Farnham (vsx8ws)
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
# Read in datasets
graduation = pd.read_csv('cc_institution_details.csv')
placement = pd.read_csv('Placement_Data_Full_Class.csv')

# %%
# STEP 1:
# QUESTIONS
# College Completion Dataset: How accurately can we predict college graduation rates?
# Campus Recruitment Dataset: How accurately can we predict post-college placement?

#%%
# STEP 2: Think of independent business metrics and prep data

#%%
# BUSINESS METRICS
# College Completion Dataset: Average difference between predicted graduation rate and real graduation rate.
# Campus Recruitment Dataset: Difference between predicted placement proportion prediction and real placement proportion.

#%%
# Check data types of graduation
graduation.info()

#%%
# Check data types of placement
placement.info()

# %%
# Look for columns with most null values (graduation)
missing_percentage_g = graduation.isnull().mean() * 100
missing_percentage_sorted_g = missing_percentage_g.sort_values(ascending=False)
print("Percentage of missing values per column (graduation):")
print(missing_percentage_sorted_g.head(60))

# %%
# Look for columns with the most null values (placement)
missing_percentage_p = placement.isnull().mean() * 100
missing_percentage_sorted_p = missing_percentage_p.sort_values(ascending=False)
print("Percentage of missing values per column (placement):")
print(missing_percentage_sorted_p.head())

# %%
# Drop columns in graduation dataset with more than 90% missing
graduation = graduation.drop(columns=missing_percentage_sorted_g[missing_percentage_sorted_g > 90].index)
graduation.info()

# %%
# Drop 'salary' column from placement dataset because it is the only column with any null values
placement = placement.drop(columns=['salary'])
placement.info()

# %%
# Convert categorical columns to the 'category' data type
categorical_cols_g = [
    'chronname',    # College/university name
    'city',         # City name
    'state',        # State
    'level',        # Two-year or four-year
    'control',      # Public or private
    'basic',        # Boolean flag
    'similar'       # Boolean flag
]
graduation[categorical_cols_g] = graduation[categorical_cols_g].astype('category')
graduation.dtypes

# %%
# Do the same for the placement dataset
categorical_cols_p = [
    'gender',        # Male or Female
    'ssc_b',         # Board of 10th grade (Central / Others)
    'hsc_b',         # Board of 12th grade (Central / Others)
    'hsc_s',         # Specialization in 12th grade
    'degree_t',      # Field of undergrad degree
    'workex',        # Yes/No for work experience
    'specialisation',# MBA specialization
    'status'         # Placed / Not placed
]
placement[categorical_cols_p] = placement[categorical_cols_p].astype('category')
placement.dtypes

# %%
# Let's look at the distribution between public and private schools
print(graduation.control.value_counts())
# Most frequent category is public, but the two types of private schools together outnumber public

# %%
# Let's look at the distribution between two-year and four-year colleges
print(graduation.level.value_counts())
# 4-year wtih a 2339 to 1459 advantage over 2-year

# %%
# Let's look at the distribution between male and female in the placement dataset
print(placement.gender.value_counts())
# Almost double the male data as compared to female

# %%
# Let's look at the distribution between work expereince in the placement dataset
print(placement.workex.value_counts())
# Almost double the amount of students with no work experience in this dataset

# %%
# Let's finally look at the distribution between placed and not placed
print(placement.status.value_counts())
# About 69% of data shows placement

# %%
