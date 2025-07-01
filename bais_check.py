import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load UCI Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('income', axis=1))
le = LabelEncoder()
df_encoded['income'] = le.fit_transform(df['income'])  # 1: >50K, 0: <=50K

# Train-test split
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# IMPORTANT: Reset index of X_test and y_test to align with predictions
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Train original model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate original model
print("=== Original Model ===")
print("Overall Accuracy:", accuracy_score(y_test, y_pred))
sex_col = 'sex_Male'
male_mask = X_test[sex_col] == 1
female_mask = X_test[sex_col] == 0
print("Male Accuracy:", accuracy_score(y_test[male_mask], y_pred[male_mask]))
print("Female Accuracy:", accuracy_score(y_test[female_mask], y_pred[female_mask]))

# Mitigate bias via oversampling females
train_df = X_train.copy()
train_df['income'] = y_train
male = train_df[train_df['sex_Male'] == 1]
female = train_df[train_df['sex_Male'] == 0]
female_upsampled = resample(female, replace=True, n_samples=len(male), random_state=42)
train_balanced = pd.concat([male, female_upsampled])
X_train_bal = train_balanced.drop('income', axis=1)
y_train_bal = train_balanced['income']

# Retrain with balanced data
model_bal = LogisticRegression(max_iter=1000)
model_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = model_bal.predict(X_test)

# Evaluate balanced model
print("\n=== Balanced Model After Mitigation ===")
print("Overall Accuracy:", accuracy_score(y_test, y_pred_bal))
print("Male Accuracy:", accuracy_score(y_test[male_mask], y_pred_bal[male_mask]))
print("Female Accuracy:", accuracy_score(y_test[female_mask], y_pred_bal[female_mask]))