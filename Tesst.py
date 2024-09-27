#Import necessary Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing #Use fetch_california_housing instead of lo

#Load the California housing dataset
housing = fetch_california_housing()
#Access the data and target variables
X= housing.data
y = housing.target
# Create a DataFrame for better readability
df = pd.DataFrame(X, columns=housing.feature_names)
