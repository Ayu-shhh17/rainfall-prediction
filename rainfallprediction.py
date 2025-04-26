# rainfall_prediction.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('rainfall.csv')  # Make sure the dataset is in the same folder

# Basic information
print("Dataset Shape:", data.shape)
print("Dataset Columns:", data.columns)
print(data.head())

# Handling missing values
data = data.dropna()

# Feature and target selection
X = data.drop('RainTomorrow', axis=1)  # Assuming 'RainTomorrow' is the target variable
y = data['RainTomorrow']

# Encode categorical variables if any
X = pd.get_dummies(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions and evaluation
log_pred = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, log_pred))

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report (Random Forest):\n", classification_report(y_test, rf_pred))

# Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
