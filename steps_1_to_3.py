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
# Let's test to see if the categorical variable transformation worked

# %%
# Distribution between public and private schools
print(graduation.control.value_counts())
# Most frequent category is public, but the two types of private schools together outnumber public

# %%
# Distribution between two-year and four-year colleges
print(graduation.level.value_counts())
# 4-year wtih a 2339 to 1459 advantage over 2-year

# %%
# Distribution between male and female in the placement dataset
print(placement.gender.value_counts())
# Almost double the male data as compared to female

# %%
# Distribution between work expereince in the placement dataset
print(placement.workex.value_counts())
# Almost double the amount of students with no work experience in this dataset

# %%
# Now I will work only with only the graduation dataset before moving on to the placement dataset

# %%
# grad_150_value will be my target variable
# This variable shows the proportion of graduation after 6 years
y_grad = graduation['grad_150_value']

# %%
# Drop rows with missing graduation rate
graduation = graduation.dropna(subset=['grad_150_value'])

# %%
# Separate my features and my target
y_grad = graduation['grad_150_value'] # model target
X_grad = graduation.drop(columns=['grad_150_value']) # preprocessing

# %%
column_list = graduation.columns.tolist()
print(column_list)

# %%
# Columns to remove before modeling graduation rate
cols_to_drop = [

    # no predictive value
    'index',       
    'unitid',   
    'chronname',    
    'city',         
    'site',        

    # geographic coordinates
    'long_x',       
    'lat_y',       

    # percentile-based features dropped to avoid redudancy 
    'exp_award_percentile',     
    'fte_percentile',
    'med_sat_percentile',
    'aid_percentile',
    'endow_percentile',
    'grad_100_percentile',
    'grad_150_percentile',
    'pell_percentile',
    'retain_percentile',
    'ft_fac_percentile'
]

# Drop the columns
X_grad = X_grad.drop(columns=cols_to_drop)

# %%
# gather numeric columns
numeric_cols = X_grad.select_dtypes(include='number').columns

# min-max seems good for default
scaler = MinMaxScaler()
X_grad[numeric_cols] = scaler.fit_transform(X_grad[numeric_cols])

# %%
# one-hot encode categorical columns
categorical_cols = X_grad.select_dtypes(include='category').columns

X_grad = pd.get_dummies(
    X_grad,
    columns=categorical_cols,
    drop_first=True
)

# %%
# train/tune/test split
# Train vs temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_grad, y_grad, test_size=0.30, random_state=42
)

# Tune vs test
X_tune, X_test, y_tune, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
# 70% train, 15% tune, 15% test (google said this was a common split)

# %%
# Now we will work on the placement dataset

# %%
# Target variable: placed (tells us either placed or not placed)
# Convert variable to binary
placement['placed'] = (placement.status == 'Placed').astype(int)

# %%
# Calculate prevalance of target variable
prevalence_place = placement.placed.mean()
print(f"Placement prevalence (baseline): {prevalence_place:.2%}")
# We find a 68.84% prevalence

# %%
# Drop unneeded columns
placement_clean = placement.drop(columns=[
    'sl_no',    # Serial number has no predictive value
    'status'    # target variable
])

# %%
# Normalize numeric columns with min-max
numeric_cols_p = placement_clean.select_dtypes(include='number').columns.tolist()
numeric_cols_p.remove('placed')

scaler_p = MinMaxScaler()
placement_clean[numeric_cols_p] = scaler_p.fit_transform(
    placement_clean[numeric_cols_p]
)

# %%
# One-hot encode categorical variables
categorical_cols_p = placement_clean.select_dtypes('category').columns.tolist()

placement_encoded = pd.get_dummies(
    placement_clean,
    columns=categorical_cols_p,
    drop_first=True
)

# %%
# train/tune/test split
# Train vs Test
train_p, temp_p = train_test_split(
    placement_encoded,
    train_size=0.7,
    stratify=placement_encoded.placed,
    random_state=42
)

# Tune vs Test
tune_p, test_p = train_test_split(
    temp_p,
    train_size=0.5,
    stratify=temp_p.placed,
    random_state=42
)

# %%
# STEP 3

# Graduation Dataset:
# This dataset seems well-suited for predicting graduation rates because 
# of data regarding enrollment size, financial resources, student aid, and retention.
# Although, becuase this dataset has 63 total columns, I worry that I was too quick
# to remove some of the features that could have been helpful in order to reduce complexity.
# There is a lot more data in this dataset than the placement dataset and I worry that I 
# haven't considered all of my options for this dataset as well as the other.
# I also think predicting a value rather than a binary variable adds a level of complexity 
# to the predictions in a way where faults in my inferences could cause wildly inacurate results.

# Placement Dataset:
# I think that this dataset is appropriate for predicting placement outcomes 
# because it includes academic performance, specialization, and work experience.
# I am worried that the dataset is too small to avoid invalid variance. 
# I am also worried that there is a gender imbalance as there is almost twice as
# much male data as there is female data.
# I also dropped the salary column for simplicity sake because it was the only 
# column with null values, but I am worried that I may be missing out on a good predictor
# due to that decision.
