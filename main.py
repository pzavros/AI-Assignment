import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import recall_score

# Load Data
cancerData = pd.read_csv("datasets/Cancer_Data.csv")

# histogram
plt.figure(figsize=(30, 20)) 
cancerData.hist(bins=15, layout=(8, 4), figsize=(30, 20))
plt.tight_layout()
plt.show()

# Preprocessing
# Count missing values before imputation
print("Missing values per column before imputation:")
print(cancerData.isna().sum())

# Handle missing values
# Separate numeric and non-numeric columns
numeric_cols = cancerData.select_dtypes(include=[np.number]).columns
non_numeric_cols = cancerData.select_dtypes(exclude=[np.number]).columns

# Apply imputation only to numeric columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
cancerData_numeric_imputed = pd.DataFrame(imputer.fit_transform(cancerData[numeric_cols]), columns=numeric_cols)

# Reconstruct the DataFrame with both numeric and non-numeric columns
cancerData_imputed = pd.concat([cancerData[non_numeric_cols], cancerData_numeric_imputed], axis=1)

# Encoding the Categorical Data
# Encoding the 'diagnosis' column using Label Encoding
label_encoder = LabelEncoder()
cancerData_imputed['diagnosis_encoded'] = label_encoder.fit_transform(cancerData_imputed['diagnosis'])

# Display the original and encoded diagnosis values
print("Original and Encoded 'diagnosis' values:")
print(cancerData_imputed[['diagnosis', 'diagnosis_encoded']].head(10))

# Feature Scaling - Scale only the numeric columns
scaler = StandardScaler()
cancerData_imputed[numeric_cols] = scaler.fit_transform(cancerData_imputed[numeric_cols])

# Display changes after scaling
print("Data after feature scaling:")
print(cancerData_imputed[numeric_cols].head(10))

# Count missing values after imputation
print("Missing values per column after imputation:")
print(cancerData_imputed.isna().sum())

# Splitting the dataset into the Training set and Test set
X = cancerData_imputed[numeric_cols].values
y = cancerData_imputed['diagnosis_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Logistic Regression model on the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the results on Training and Test sets
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

# Computing the accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate recall for the training set
train_recall = recall_score(y_train, y_train_pred)

print(f"Training Set Recall: {train_recall}")

print(f"Training Set Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")

# Making the Confusion Matrix for the Test set
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for the Test Set:")
print(cm)

# Computing recall for the Test set
test_recall = recall_score(y_test, y_test_pred)
print(f"Test Set Recall: {test_recall}")

# Optionally, print predicted probabilities for the Test set
y_pred_proba = classifier.predict_proba(X_test)[:, 1]
print("Predicted probabilities for the Test Set:")
print(y_pred_proba)
