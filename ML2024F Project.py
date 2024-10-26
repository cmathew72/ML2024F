import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

# One-hot encode categorical variables
train_data_encoded = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

# Align columns of test data with training data
missing_cols = set(train_data_encoded.columns) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0
test_data_encoded = test_data_encoded[train_data_encoded.columns.drop('income>50K')]

# Split features and target
X = train_data_encoded.drop(columns=['income>50K'])
y = train_data_encoded['income>50K']

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data_encoded)

# Split into training and validation sets for meta-model training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize base models with fine-tuned parameters
xgb_model = xgb.XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=1000, subsample=0.9, colsample_bytree=0.8, eval_metric='auc', random_state=42)
rf_model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5, random_state=42)
logreg_model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)

# Use KFold for out-of-fold predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for meta-features
train_meta = np.zeros((X_train.shape[0], 3))  # 3 models
test_meta = np.zeros((test_data_scaled.shape[0], 3))

# Loop through each fold for stacking
for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # XGBoost
    xgb_model.fit(X_train_fold, y_train_fold)
    train_meta[val_idx, 0] = xgb_model.predict_proba(X_val_fold)[:, 1]
    test_meta[:, 0] += xgb_model.predict_proba(test_data_scaled)[:, 1] / kf.n_splits

    # Random Forest
    rf_model.fit(X_train_fold, y_train_fold)
    train_meta[val_idx, 1] = rf_model.predict_proba(X_val_fold)[:, 1]
    test_meta[:, 1] += rf_model.predict_proba(test_data_scaled)[:, 1] / kf.n_splits

    # Logistic Regression
    logreg_model.fit(X_train_fold, y_train_fold)
    train_meta[val_idx, 2] = logreg_model.predict_proba(X_val_fold)[:, 1]
    test_meta[:, 2] += logreg_model.predict_proba(test_data_scaled)[:, 1] / kf.n_splits

# Train the meta-model (Logistic Regression) on the out-of-fold predictions
meta_model = LogisticRegression()
meta_model.fit(train_meta, y_train)

# Validate on validation set
y_val_pred_meta = meta_model.predict_proba(train_meta)[:, 1]
auc_meta = roc_auc_score(y_train, y_val_pred_meta)
print(f"Stacked Model AUC on Validation Set: {auc_meta}")

# Predict on the test set
test_predictions_meta = meta_model.predict_proba(test_meta)[:, 1]

# Prepare the submission file
submission_meta = pd.DataFrame({
    'ID': test_data['ID'],  # Assuming the test set has an 'ID' column
    'Prediction': test_predictions_meta
})

# Save the submission file
submission_meta.to_csv('submission.csv', index=False)
