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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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



#BALANCE THE DATASET !!!!!!!!!!!!!!!!!!!!

#PIPELINE KFOLD CROSS VALIDATION
#IF !PIPELINE, CREATE NEW CSV 
#BALANCE ONLY THE TRAINING
#PRECISION = TRUE POSITIVE/TRUE POSITIVIE + FALSE POSITIVE
#HYPER PARAMETERS/ TUNING PARAMETERS

#RANDOMIZE THE DATASET BEFORE SPLITTING !!!!!!!!!!!!!!!!!!


# Splitting the dataset into the Training set and Test set 
X = cancerData_imputed[numeric_cols].values
y = cancerData_imputed['diagnosis_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# import SMOTE module from imblearn library
# pip install imblearn (if you don't have imblearn in your system already)


# Training the Logistic Regression model on the Training set
# RANDOM STATE = 42 ALWAYS CONSTANT
# DOCUMENTATION
# decision tree genie, kati allo
classifier = LogisticRegression(C=0.1,random_state=0)
classifier2 = LogisticRegression(C=0.01,random_state=0)
classifier.fit(X_train, y_train)
# We tested this and we didn't like it "classifier = LogisticRegression(C=0.01, penalty='l2', solver='saga', random_state=0)"

#Logistic Regression:
#Training Set Accuracy: 98.9010989010989
#Test Set Accuracy: 96.49122807017544
#Training Set Recall: 97.57575757575758
#Test Set Recall: 95.74468085106383


# Training the Random Forest model on the Training set
#NUMBER OF TREES
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train, y_train)

# Predicting the results on Training and Test sets for both models
y_train_pred_lr = classifier.predict(X_train)
y_test_pred_lr = classifier.predict(X_test)
y_train_pred_rf = rf_classifier.predict(X_train)
y_test_pred_rf = rf_classifier.predict(X_test)

# Computing the accuracies and recalls
train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
train_recall_lr = recall_score(y_train, y_train_pred_lr)
test_recall_lr = recall_score(y_test, y_test_pred_lr)

train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
train_recall_rf = recall_score(y_train, y_train_pred_rf)
test_recall_rf = recall_score(y_test, y_test_pred_rf)

# Printing the results
print("Logistic Regression:")
print(f"Training Set Accuracy: {train_accuracy_lr*100}")
print(f"Test Set Accuracy: {test_accuracy_lr*100}")
print(f"Training Set Recall: {train_recall_lr*100}")
print(f"Test Set Recall: {test_recall_lr*100}")

print("Random Forest:")
print(f"Training Set Accuracy: {train_accuracy_rf*100}")
print(f"Test Set Accuracy: {test_accuracy_rf*100}")
print(f"Training Set Recall: {train_recall_rf*100}")
print(f"Test Set Recall: {test_recall_rf*100}")

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_test_pred_lr)
print("Confusion Matrix for Logistic Regression:")
print(cm_lr)

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
print("Confusion Matrix for Random Forest:")
print(cm_rf)

# Training the SVM model on the Training set

#Linear Algorithm
ln_classifier = SVC(kernel = 'linear', random_state = 0)
ln_classifier.fit(X_train, y_train)

#Apply the algorithm on train
y_train_pred_ln = ln_classifier.predict(X_train)
y_test_pred_ln = ln_classifier.predict(X_test)

#Find the accuracy of linear
train_accuracy_ln = accuracy_score(y_train, y_train_pred_ln)
test_accuracy_ln = accuracy_score(y_test, y_test_pred_ln)

#Find the recall of linear
train_recall_ln = recall_score(y_train, y_train_pred_ln)
test_recall_ln = recall_score(y_test, y_test_pred_ln)

print("Linear Algorithm:")
print(f"Training Set Accuracy: {train_accuracy_ln*100}")
print(f"Test Set Accuracy: {test_accuracy_ln*100}")
print(f"Training Set Recall: {train_recall_ln*100}")
print(f"Test Set Recall: {test_recall_ln*100}")
