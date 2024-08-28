# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Read in the dataset
loans = pd.read_csv("loans.csv")

# Preview the data
loans.head()

# Remove the loan_id to avoid accidentally using it as a feature
loans.drop(columns=["loan_id"], inplace=True)

# Counts and data types per column
loans.info()

# Distributions and relationships
sns.pairplot(data=loans, diag_kind="kde", hue="loan_status")
plt.show()

# Correlation between variables
sns.heatmap(loans.corr(), annot=True)
plt.show()

# Target frequency
loans["loan_status"].value_counts(normalize=True)

# Class frequency by loan_status
for col in loans.columns[loans.dtypes == "object"]:
    sns.countplot(data=loans, x=col, hue="loan_status")
    plt.show()

# First model using loan_amount
x = loans[["loan_amount"]]
y = loans[["loan_status"]]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Previewing the training set
print(X_train[:5], "\n", y_train[:5])

# Instantiate a logistic regression model
clf = LogisticRegression(random_state=42)

# Fit to the training data
clf.fit(X_train, y_train)

# Predict test set values
y_pred = clf.predict(X_test)

# Check the model's first five predictions
print(y_pred[:5])

# Accuracy
print(clf.score(X_test, y_test))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Rejected", "Approved"])
disp.plot()
plt.show()

# Convert categorical features to binary
loans = pd.get_dummies(loans)

# Previewing the new DataFrame
loans.head()

# Resplit into features and targets
X = loans.drop(columns=["loan_status"])
y = loans[["loan_status"]]
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instantiate logistic regression model
clf = LogisticRegression(random_state=42)
# Fit to the training data
clf.fit(X_train, y_train)
# Predict test set values
y_pred = clf.predict(X_test)

# Accuracy
print(clf.score(X_test, y_test))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Rejected", "Approved"])
disp.plot()
plt.show()

# Finding the importance of features
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": clf.coef_.reshape(-1)
})
plt.figure(figsize=(9,5))
sns.barplot(data=feature_importance, x="feature", y="importance")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
