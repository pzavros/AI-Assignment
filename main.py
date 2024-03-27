import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV

# Load Data
cancerData = pd.read_csv("datasets/Cancer_Data.csv")

# Histogram of the data
plt.figure(figsize=(30, 20))
cancerData.hist(bins=15, layout=(8, 4), figsize=(30, 20))
plt.tight_layout()
plt.show()

# Preprocessing
print("\nMissing values per column before imputation:")
print(cancerData.isna().sum())

numeric_cols = cancerData.select_dtypes(include=[np.number]).columns
non_numeric_cols = cancerData.select_dtypes(exclude=[np.number]).columns

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
cancerData_numeric_imputed = pd.DataFrame(imputer.fit_transform(cancerData[numeric_cols]), columns=numeric_cols)
cancerData_imputed = pd.concat([cancerData[non_numeric_cols], cancerData_numeric_imputed], axis=1)

label_encoder = LabelEncoder()
cancerData_imputed['diagnosis_encoded'] = label_encoder.fit_transform(cancerData_imputed['diagnosis'])

scaler = StandardScaler()
cancerData_imputed[numeric_cols] = scaler.fit_transform(cancerData_imputed[numeric_cols])

print("\nData after feature scaling:")
print(cancerData_imputed[numeric_cols].head(10))

# Splitting the dataset into Training set and Test set
X = cancerData_imputed[numeric_cols].values
y = cancerData_imputed['diagnosis_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Balancing the Training Dataset with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Hyperparameter Tuning with GridSearchCV for Logistic Regression, Random Forest, and SVC
# Logistic Regression
lr = LogisticRegression(random_state=0)
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train_balanced, y_train_balanced)

# Random Forest
rf = RandomForestClassifier(random_state=0)
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_balanced, y_train_balanced)

# Support Vector Machine
svc = SVC(random_state=0)
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svc.fit(X_train_balanced, y_train_balanced)

# Print best parameters and scores
print("\nBest parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best score for Logistic Regression:", grid_search_lr.best_score_)
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best score for Random Forest:", grid_search_rf.best_score_)
print("Best parameters for SVC:", grid_search_svc.best_params_)
print("Best score for SVC:", grid_search_svc.best_score_)

# Function to print metrics and confusion matrix
def print_metrics_and_confusion_matrix(classifier, X_test, y_test, classifier_name):
    y_pred = classifier.predict(X_test)
    print(f"\n{classifier_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
    print("Confusion Matrix:")
    
# Confusion matrix, example representation
#     [[True Negatives (TN)  False Positives (FP)]
#     [False Negatives (FN) True Positives (TP)]]
    print(confusion_matrix(y_test, y_pred))

# Use best parameters for training
lr_classifier = LogisticRegression(**grid_search_lr.best_params_, random_state=0)
rf_classifier = RandomForestClassifier(**grid_search_rf.best_params_, random_state=0)
svc_classifier = SVC(**grid_search_svc.best_params_, random_state=0)

# ------------------------ TRAIN ON UNBALANCED DATA ---------------------------
lr_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)
svc_classifier.fit(X_train, y_train)

# Evaluation
print("\n ----- Unbalanced Data ----- ")
print_metrics_and_confusion_matrix(lr_classifier, X_test, y_test, "Logistic Regression")
print_metrics_and_confusion_matrix(rf_classifier, X_test, y_test, "Random Forest")
print_metrics_and_confusion_matrix(svc_classifier, X_test, y_test, "SVC")

# ------------------------ TRAIN ON BALANCED DATA ---------------------------
lr_classifier_balanced = LogisticRegression(**grid_search_lr.best_params_, random_state=0)
rf_classifier_balanced = RandomForestClassifier(**grid_search_rf.best_params_, random_state=0)
svc_classifier_balanced = SVC(**grid_search_svc.best_params_, random_state=0)

lr_classifier_balanced.fit(X_train_balanced, y_train_balanced)
rf_classifier_balanced.fit(X_train_balanced, y_train_balanced)
svc_classifier_balanced.fit(X_train_balanced, y_train_balanced)

# Evaluation
print("\n ----- Balanced Data ----- ")
print_metrics_and_confusion_matrix(lr_classifier_balanced, X_test, y_test, "Logistic Regression")
print_metrics_and_confusion_matrix(rf_classifier_balanced, X_test, y_test, "Random Forest")
print_metrics_and_confusion_matrix(svc_classifier_balanced, X_test, y_test, "SVC")

