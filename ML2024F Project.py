import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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

# Align columns of test data with training data as before
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

# -------------------------- XGBoost Hyperparameter Tuning with Grid Search --------------------------

# Define parameter grid for XGBoost
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='auc', random_state=42)

# Perform grid search with 5-fold cross-validation
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_xgb.fit(X_train_scaled, y_train)

# Get the best parameters and the best AUC score
best_params_xgb = grid_search_xgb.best_params_
best_auc_xgb = grid_search_xgb.best_score_

print(f"Best XGBoost Parameters: {best_params_xgb}")
print(f"Best XGBoost AUC from Grid Search: {best_auc_xgb}")

# Refit XGBoost with the best parameters
xgb_best_model = xgb.XGBClassifier(**best_params_xgb, eval_metric='auc', random_state=42)
xgb_best_model.fit(X_train_scaled, y_train)

# Predict probabilities for the validation set
y_val_pred_proba_xgb_best = xgb_best_model.predict_proba(X_val_scaled)[:, 1]

# Evaluate the tuned XGBoost model on the validation set
auc_xgb_best = roc_auc_score(y_val, y_val_pred_proba_xgb_best)
print(f"Tuned XGBoost AUC on Validation Set: {auc_xgb_best}")

# Predict for the test set
test_predictions_proba_xgb_best = xgb_best_model.predict_proba(test_data_scaled)[:, 1]

# Prepare the submission file
submission_xgb_best = pd.DataFrame({
    'ID': test_data['ID'],  # Assuming the test set has an 'ID' column
    'Prediction': test_predictions_proba_xgb_best
})

# Save the tuned XGBoost submission file
submission_xgb_best.to_csv('submission.csv', index=False)