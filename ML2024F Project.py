import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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

# Define categorical columns for one-hot encoding
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# Feature Engineering

# Binning continuous variables: Create age groups and hours worked per week bins
train_data['age_binned'] = pd.cut(train_data['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=[1, 2, 3, 4, 5, 6])
test_data['age_binned'] = pd.cut(test_data['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=[1, 2, 3, 4, 5, 6])

train_data['hours_per_week_binned'] = pd.cut(train_data['hours.per.week'], bins=[0, 20, 40, 60, 80, 100], labels=[1, 2, 3, 4, 5])
test_data['hours_per_week_binned'] = pd.cut(test_data['hours.per.week'], bins=[0, 20, 40, 60, 80, 100], labels=[1, 2, 3, 4, 5])

# Interaction terms: Create interaction features between workclass and education
train_data['workclass_education'] = train_data['workclass'] + '_' + train_data['education']
test_data['workclass_education'] = test_data['workclass'] + '_' + test_data['education']

# One-hot encode categorical variables and new interaction terms
categorical_columns_extended = categorical_columns + ['workclass_education', 'age_binned', 'hours_per_week_binned']
train_data_encoded = pd.get_dummies(train_data, columns=categorical_columns_extended, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=categorical_columns_extended, drop_first=True)

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

# -------------------------- Random Forest Model with Hyperparameter Tuning --------------------------

# Define a parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search with 5-fold cross-validation
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)

# Get the best parameters and AUC score
best_params_rf = grid_search_rf.best_params_
best_auc_rf = grid_search_rf.best_score_

print(f"Best Random Forest Parameters: {best_params_rf}")
print(f"Best Random Forest AUC: {best_auc_rf}")

# Refit the Random Forest model with the best parameters
rf_best_model = RandomForestClassifier(**best_params_rf, random_state=42)
rf_best_model.fit(X_train_scaled, y_train)

# Predict probabilities for the validation set
y_val_pred_proba_rf = rf_best_model.predict_proba(X_val_scaled)[:, 1]

# Evaluate the tuned Random Forest model using AUC
auc_rf = roc_auc_score(y_val, y_val_pred_proba_rf)
print(f"Tuned Random Forest AUC: {auc_rf}")

# Predict for the test set
test_predictions_proba_rf = rf_best_model.predict_proba(test_data_scaled)[:, 1]

# Prepare the submission file for Random Forest
submission_rf = pd.DataFrame({
    'ID': test_data['ID'],  # Assuming the test set has an 'ID' column
    'Prediction': test_predictions_proba_rf
})

# -------------------------- XGBoost Model --------------------------

# Initialize and train an XGBoost model without the 'use_label_encoder' parameter
xgb_model = xgb.XGBClassifier(eval_metric='auc', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predict probabilities for the validation set
y_val_pred_proba_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]

# Evaluate the XGBoost model using AUC
auc_xgb = roc_auc_score(y_val, y_val_pred_proba_xgb)
print(f"XGBoost AUC: {auc_xgb}")

# Predict for the test set
test_predictions_proba_xgb = xgb_model.predict_proba(test_data_scaled)[:, 1]

# Prepare the submission file for XGBoost
submission_xgb = pd.DataFrame({
    'ID': test_data['ID'],  # Assuming the test set has an 'ID' column
    'Prediction': test_predictions_proba_xgb
})

# Save the XGBoost submission file
submission_xgb.to_csv('submission.csv', index=False)