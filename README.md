Kyphosis Dataset - Decision Tree & Random Forest Classifiers
Overview
This project focuses on building machine learning models to classify the presence of Kyphosis, a medical condition related to the curvature of the spine, using Decision Tree and Random Forest algorithms.

Models:
Decision Tree Classifier: A tree-based algorithm used to model decision paths and outcomes.
Random Forest Classifier: An ensemble learning method that combines multiple decision trees to improve accuracy and avoid overfitting.
Dataset
The Kyphosis dataset is a small dataset provided by Scikit-learn, which includes information about children who have undergone spinal surgery. The target variable indicates whether kyphosis is present after the surgery.

Features:
Kyphosis: Whether the kyphosis condition is present (absent or present).
Age: Age of the patient in months.
Number: Number of vertebrae involved in the operation.
Start: The number of the first (topmost) vertebra that was operated on.
Steps to Follow
1. Set up the Environment
To get started, ensure the following packages are installed:

Python 3.x
Libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
You can install the required libraries using:

bash
نسخ الكود
pip install numpy pandas matplotlib seaborn scikit-learn
2. Load and Inspect the Data
First, load the Kyphosis dataset, inspect the data, and visualize it.

python
نسخ الكود
import pandas as pd
from sklearn.datasets import load_kyphosis

# Load the dataset
data = load_kyphosis(as_frame=True)
df = data['frame']

# Display the first few rows
df.head()
3. Exploratory Data Analysis (EDA)
Visualize the data to understand the relationships between features.

python
نسخ الكود
import seaborn as sns
import matplotlib.pyplot as plt

# Plot pairplot for data visualization
sns.pairplot(df, hue="Kyphosis")
plt.show()
4. Data Preprocessing
Before building models, encode the target variable and split the dataset into training and testing sets.

python
نسخ الكود
from sklearn.model_selection import train_test_split

# Split features (X) and target (y)
X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
5. Building the Decision Tree Model
Fit a Decision Tree classifier and evaluate its performance.

python
نسخ الكود
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Make predictions
dtree_pred = dtree.predict(X_test)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test, dtree_pred))
print(classification_report(y_test, dtree_pred))
6. Building the Random Forest Model
Train a Random Forest classifier, which is an ensemble of decision trees.

python
نسخ الكود
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# Make predictions
rfc_pred = rfc.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
7. Model Evaluation
Compare the performance of both models:

Accuracy: Measures the percentage of correctly classified instances.
Precision: Indicates the proportion of true positive results.
Recall: Measures the proportion of actual positives that are correctly identified.
F1-Score: Combines precision and recall into a single metric.
8. Visualize Decision Tree
You can visualize the structure of the trained Decision Tree for better interpretability.

python
نسخ الكود
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=data['target_names'])
plt.show()
9. Feature Importance (Random Forest)
Check which features are most important in predicting kyphosis using the Random Forest model.

python
نسخ الكود
import numpy as np

# Get feature importances
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for i in range(X.shape[1]):
    print(f"{i + 1}. feature {X.columns[indices[i]]} ({importances[indices[i]]})")
10. Save and Export the Model (Optional)
You can save the trained models for future use using joblib or pickle.

python
نسخ الكود
import joblib

# Save Random Forest model
joblib.dump(rfc, 'random_forest_model.pkl')

# Save Decision Tree model
joblib.dump(dtree, 'decision_tree_model.pkl')
Conclusion
This project demonstrates how to build, train, and evaluate Decision Tree and Random Forest classifiers using the Kyphosis dataset. Random Forest, as an ensemble model, tends to provide more stable and accurate results, but the Decision Tree model offers better interpretability.

Requirements
Python 3.x
Libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
