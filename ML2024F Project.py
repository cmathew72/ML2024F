import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Load the datasets
train_data = pd.read_csv('train_final.csv')
test_data = pd.read_csv('test_final.csv')

# Preprocessing: Replace '?' with NaN and fill missing values
train_data.replace('?', pd.NA, inplace=True)
test_data.replace('?', pd.NA, inplace=True)

# Fill missing values with the most frequent value
for column in ['workclass', 'occupation', 'native.country']:
    most_frequent_value = train_data[column].mode()[0]
    train_data[column] = train_data[column].fillna(most_frequent_value)
    test_data[column] = test_data[column].fillna(most_frequent_value)

# One-hot encode categorical variables
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
train_data_encoded = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

# Align columns of test data with training data
missing_cols = set(train_data_encoded.columns) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0
test_data_encoded = test_data_encoded[train_data_encoded.columns.drop('income>50K')]  # Align with training features

# Split features and target
X = train_data_encoded.drop(columns=['income>50K'])
y = train_data_encoded['income>50K']

# Split into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_data_scaled = scaler.transform(test_data_encoded)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Penalty type
    'solver': ['liblinear']  # Required for l1 penalty
}

# Initialize logistic regression model
logreg = LogisticRegression(random_state=42, max_iter=1000)

# Set up the grid search with 5-fold cross-validation
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and the corresponding AUC score
best_params = grid_search.best_params_
best_auc = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best AUC Score: {best_auc}")