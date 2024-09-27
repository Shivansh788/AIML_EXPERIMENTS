# Write a python program to import and export data using Pandas and 
# show the details of the dataset like number of rows, columns, first five 
# rows, size, number of missing values, sum, average, min and max 
# values from the numerical columns.

import pandas as pd

# Import data from a CSV file
df = pd.read_csv('student-scores.csv')

# Display dataset details
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
# printing 5 rows 
print("\nFirst five rows:\n", df.head())
print("\nSize of the dataset:", df.size)
print("\nNumber of missing values:\n", df.isnull().sum())

# Display summary statistics for numerical columns
print("\nSum of numerical columns:\n", df.sum(numeric_only=True))
print("\nAverage of numerical columns:\n", df.mean(numeric_only=True))
print("\nMinimum values of numerical columns:\n", df.min(numeric_only=True))
print("\nMaximum values of numerical columns:\n", df.max(numeric_only=True))

# Export data to a new CSV file
df.to_csv('output.csv', index=False)