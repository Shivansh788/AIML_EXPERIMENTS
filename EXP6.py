import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('student-scores.csv')

# Handling Missing Values

# Option 1: Remove rows with missing values
df_cleaned = df.dropna()

# Option 2: Fill missing values with the mean (for numerical columns)
df_filled = df.fillna(df.mean())

# Option 3: Fill missing values with the median (for numerical columns)
df_filled_median = df.fillna(df.median())

# Option 4: Fill missing values with the mode (for categorical columns)
df_filled_mode = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'O' else x)

# Handling Outliers

# Using the Interquartile Range (IQR) method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = ((df < lower_bound) | (df > upper_bound))

# Option 1: Remove outliers
df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Option 2: Cap outliers to the lower and upper bounds
df_capped = df.copy()
df_capped[df < lower_bound] = lower_bound
df_capped[df > upper_bound] = upper_bound

# Option 3: Impute outliers with mean/median
df_imputed_outliers = df.copy()
df_imputed_outliers[outliers] = np.nan
df_imputed_outliers = df_imputed_outliers.fillna(df.mean())

print("Original DataFrame:\n", df)
print("DataFrame after handling missing values and outliers:\n", df_imputed_outliers)
