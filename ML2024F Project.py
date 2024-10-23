import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_data_scaled = scaler.transform(test_data_encoded)

# Train logistic regression
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Predict probabilities for the validation set
y_val_pred_proba = logreg.predict_proba(X_val_scaled)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC: {auc_score}")

# Predict for the test set
test_predictions_proba = logreg.predict_proba(test_data_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({
    'ID': test_data['ID'],  # Assuming the test set has an 'ID' column
    'Prediction': test_predictions_proba
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
