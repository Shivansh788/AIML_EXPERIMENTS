#1. Mean/Median/Mode Imputation
import pandas as pd
from sklearn.impute import KNNImputer

# Sample DataFrame
data = {'A': [1, 2, None, 4], 'B': [None, 2, 3, 4]}
df = pd.DataFrame(data)

# Mean Imputation
df['A'].fillna(df['A'].mean(), inplace=True)

# Median Imputation
df['B'].fillna(df['B'].median(), inplace=True)

# Mode Imputation (for categorical data)
df['C'].fillna(df['C'].mode()[0], inplace=True)

print(df)

#2. K-Nearest Neighbors (KNN) Imputation

# Sample DataFrame
data = [[1, 2], [None, 3], [7, None], [4, 5]]
imputer = KNNImputer(n_neighbors=2)
imputed_data = imputer.fit_transform(data)

print(imputed_data)

#3. Regression Imputation
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
data = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [1, 2, 3, 4]})
train_data = data[data['A'].notnull()]
X_train = train_data[['B']]
y_train = train_data['A']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing values
missing_data = data[data['A'].isnull()]
X_missing = missing_data[['B']]
data.loc[data['A'].isnull(), 'A'] = model.predict(X_missing)

print(data)

#4. Multiple Imputation
#Using fancyimpute library:

from fancyimpute import IterativeImputer

# Sample DataFrame
data = pd.DataFrame([[1, 2], [np.nan, 3], [7, np.nan], [4, 5]])

# Multiple Imputation
imputer = IterativeImputer()
imputed_data = imputer.fit_transform(data)

print(imputed_data)

#5. Last Observation Carried Forward (LOCF)
# Sample DataFrame
data = {'A': [1, 2, None, 4, None], 'B': [None, 2, 3, None, 5]}
df = pd.DataFrame(data)

# LOCF
df.fillna(method='ffill', inplace=True)

print(df)

#6. Interpolation
# Sample DataFrame
data = {'A': [1, 2, None, 4, 5]}
df = pd.DataFrame(data)

# Interpolation
df['A'] = df['A'].interpolate()

print(df)