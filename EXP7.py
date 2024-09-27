import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load your dataset
df = pd.read_csv('student-scores.csv')

# Feature Scaling
# Initialize the StandardScaler
scaler = StandardScaler()

# Select numerical columns for scaling
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Fit and transform the numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# One-Hot Encoding
# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Select categorical columns for encoding
categorical_cols = df.select_dtypes(include=['object']).columns

# Fit and transform the categorical columns
encoded_features = encoder.fit_transform(df[categorical_cols])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate the encoded features with the original DataFrame
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

print("DataFrame after feature scaling and one-hot encoding:\n", df)
