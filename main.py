import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score,  precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier

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


#-------MLP ALGORITHM-------#

ann = tf.keras.models.Sequential()

# Add the input layer and the first hidden layer with 6 units and relu activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the second hidden layer with 6 units and relu activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the output layer with 1 output unit and sigmoid activation function
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the ANN with adam optimizer, loss function = binary crossentropy, and metrics = accuracy
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN on the Training set constructed with batch size 32 and epochs 100
ann.fit(X_train, y_train, batch_size=32, epochs=100)




# Make predictions based on the test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Output the predictions alongside the actual values
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Construct the Confusion Matrix and calculate the accuracy
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# Calculate the False Positive Rate
FP_rate = FP / (FP + TN)

#Unblananced data
print(cm)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"FP Rate Score: {FP_rate*100:.2f}%")


#---------------------------------------
ann_balanced = tf.keras.models.Sequential()

# Add the input layer and the first hidden layer with 6 units and relu activation function
ann_balanced .add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the second hidden layer with 6 units and relu activation function
ann_balanced .add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the output layer with 1 output unit and sigmoid activation function
ann_balanced .add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the ANN with adam optimizer, loss function = binary crossentropy, and metrics = accuracy
ann_balanced .compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN on the Training set constructed with batch size 32 and epochs 100
ann_balanced .fit(X_train_balanced, y_train_balanced, batch_size=32, epochs=100)




# Make predictions based on the test set results
y_pred_balanced = ann_balanced .predict(X_test)
y_pred_balanced = (y_pred_balanced > 0.5)

# Output the predictions alongside the actual values
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Construct the Confusion Matrix and calculate the accuracy
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

# Calculate the False Positive Rate for Balanced Data
TN_balanced, FP_balanced, FN_balanced, TP_balanced = cm_balanced.ravel()

FP_rate_balanced = FP_balanced / (FP_balanced + TN_balanced)


#Balanced Data
print(cm_balanced)
print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred_balanced)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_balanced)*100:.2f}%")
print(f"FP Rate Score: {FP_rate_balanced*100:.2f}%")


#--------------------------#


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
# Na diksoume kai manually to linear kai poly me diaforetika parameters kai diaforetiko degree gia poly 
svc = SVC(random_state=0)
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2,3,4,9]
}

grid_search_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svc.fit(X_train_balanced, y_train_balanced)

# Naive Bayes, NO NEED for HYPERPARAMETERS TUNING DEAFAULT GOOD ENOUGH BRO

decisiontree = DecisionTreeClassifier(random_state=0)
param_grid_deecision_tree = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 5, 10, 15, 50, 30],
    'max_leaf_nodes': [5, 10, 20, 50, 100, 200, 500]
}
grid_search_decision_tree = GridSearchCV(estimator=decisiontree, param_grid=param_grid_deecision_tree, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_decision_tree.fit(X_train_balanced, y_train_balanced)
# Print best parameters and scores
print("\nBest parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best score for Logistic Regression:", grid_search_lr.best_score_)
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best score for Random Forest:", grid_search_rf.best_score_)
print("Best parameters for SVC:", grid_search_svc.best_params_)
print("Best score for SVC:", grid_search_svc.best_score_)
print("Best parameter for decision tree",grid_search_decision_tree.best_params_)
print("Best score for decision tree",grid_search_decision_tree.best_score_ )

# Function to print metrics and confusion matrix
def print_metrics_and_confusion_matrix(classifier, X_test, y_test, classifier_name):
    y_pred = classifier.predict(X_test)
    print(f"\n{classifier_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
    
    print("Confusion Matrix:")
    
# Confusion matrix, example representation
#     [[True Negatives (TN)  False Positives (FP)]
#     [False Negatives (FN) True Positives (TP)]]
    print(confusion_matrix(y_test, y_pred)) 

# Use best parameters for training
logistic_regresion_classifier = LogisticRegression(**grid_search_lr.best_params_, random_state=0)
random_forest_classifier = RandomForestClassifier(**grid_search_rf.best_params_, random_state=0)
svc_linear_classifier = SVC(**grid_search_svc.best_params_, random_state=0)
naive_classifier = GaussianNB()
decision_tree_classifier = DecisionTreeClassifier(**grid_search_decision_tree.best_params_, random_state=0)



# ------------------------ TRAIN ON UNBALANCED DATA ---------------------------
logistic_regresion_classifier .fit(X_train, y_train)
random_forest_classifier .fit(X_train, y_train)
svc_linear_classifier.fit(X_train, y_train)
naive_classifier.fit(X_train, y_train)
decision_tree_classifier.fit(X_train, y_train)

# Evaluation
print("\n ----- Unbalanced Data ----- ")
print_metrics_and_confusion_matrix(logistic_regresion_classifier , X_test, y_test, "Logistic Regression:")
print_metrics_and_confusion_matrix(random_forest_classifier , X_test, y_test, "Random Forest:")
print_metrics_and_confusion_matrix(svc_linear_classifier, X_test, y_test, "SVC:")
print_metrics_and_confusion_matrix(naive_classifier, X_test, y_test, "Naive Baysain:")
print_metrics_and_confusion_matrix(decision_tree_classifier, X_test, y_test, "Decision Tree:")


# ------------------------ TRAIN ON BALANCED DATA ---------------------------
lr_classifier_balanced = LogisticRegression(**grid_search_lr.best_params_, random_state=0)
rf_classifier_balanced = RandomForestClassifier(**grid_search_rf.best_params_, random_state=0)
svc_classifier_balanced = SVC(**grid_search_svc.best_params_, random_state=0)
naive_classifier_balanced = GaussianNB()
decision_tree_classifier_balanced = DecisionTreeClassifier(**grid_search_decision_tree.best_params_, random_state=0)



lr_classifier_balanced.fit(X_train_balanced, y_train_balanced)
rf_classifier_balanced.fit(X_train_balanced, y_train_balanced)
svc_classifier_balanced.fit(X_train_balanced, y_train_balanced)
naive_classifier_balanced.fit(X_train_balanced, y_train_balanced)
decision_tree_classifier_balanced.fit(X_train_balanced, y_train_balanced)



# Evaluation
print("\n ----- Balanced Data ----- ")
print_metrics_and_confusion_matrix(lr_classifier_balanced, X_test, y_test, "Logistic Regression:")
print_metrics_and_confusion_matrix(rf_classifier_balanced, X_test, y_test, "Random Forest:")
print_metrics_and_confusion_matrix(svc_classifier_balanced, X_test, y_test, "SVC:")
print_metrics_and_confusion_matrix(naive_classifier_balanced, X_test, y_test, "Naive Baysain:")
print_metrics_and_confusion_matrix(decision_tree_classifier_balanced, X_test, y_test, "Decision Tree:")

