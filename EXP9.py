# 1. Resampling Techniques
# a. Oversampling the Minority Class:

# Random Oversampling: Duplicate examples from the minority class.
# SMOTE (Synthetic Minority Over-sampling Technique): Generate synthetic samples by interpolating between existing minority class examples.
# ADASYN (Adaptive Synthetic Sampling): Similar to SMOTE but focuses on harder-to-learn examples.
# b. Undersampling the Majority Class:

# Random Undersampling: Remove examples from the majority class.
# NearMiss: Selects examples from the majority class that are closest to the minority class examples.
# 2. Ensemble Techniques
# a. Balanced Random Forest:

# Combines the predictions of multiple decision trees, each trained on a balanced subset of the data.
# b. EasyEnsemble and BalanceCascade:

# Create multiple balanced subsets of the data and train separate models on each subset.
# 3. Algorithm-Level Methods
# a. Cost-Sensitive Learning:

# Modify the learning algorithm to pay more attention to the minority class by assigning higher misclassification costs to minority class errors.
# 4. Data Augmentation
# Create additional data points by applying transformations (e.g., rotations, translations) to existing data.
# 5. Anomaly Detection
# Treat the minority class as anomalies and use anomaly detection algorithms to identify them.
# 6. Evaluation Metrics
# Use metrics that are more informative for imbalanced data, such as Precision-Recall Curve, F1 Score, and ROC-AUC.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import BalancedRandomForestClassifier

# Load your dataset
df = pd.read_csv('student-scores.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Oversampling using SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# 2. Undersampling using NearMiss
nearmiss = NearMiss()
X_nearmiss, y_nearmiss = nearmiss.fit_resample(X_train, y_train)

# 3. Balanced Random Forest
brf = BalancedRandomForestClassifier(random_state=42)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)

# Evaluate the model
print("Balanced Random Forest Classifier Report:\n", classification_report(y_test, y_pred_brf))

# 4. Cost-Sensitive Learning
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Cost-Sensitive Random Forest Classifier Report:\n", classification_report(y_test, y_pred_rf))