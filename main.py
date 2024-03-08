import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
#data visualisation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load Data
cancerData = pd.read_csv("datasets/Cancer_Data.csv")
breastData = pd.read_csv("datasets/breast-cancer-dataset.csv")
patientData = pd.read_csv("datasets/cancer patient data sets.csv")


# histogram
plt.figure(figsize=(30, 20)) 
# Adjust the layout to prevent overlapping
cancerData.hist(bins=15, layout=(8, 4), figsize=(30, 20))
plt.tight_layout()

plt.show()


# sellect all
dep_var = cancerData.iloc[:,:].values 


#preproccesing
# Count missing values before imputation
print("Missing values per column before imputation:")
print(cancerData.isna().sum())

# Plot histograms
plt.figure(figsize=(30, 20))
cancerData.hist(bins=15, layout=(8, 4), figsize=(30, 20))
plt.tight_layout()
plt.show()

# Preprocessing
# Handle missing values
# Separate numeric and non-numeric columns
numeric_cols = cancerData.select_dtypes(include=[np.number]).columns
non_numeric_cols = cancerData.select_dtypes(exclude=[np.number]).columns

# Apply imputation only to numeric columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
cancerData_numeric_imputed = pd.DataFrame(imputer.fit_transform(cancerData[numeric_cols]), columns=numeric_cols)

# Reconstruct the DataFrame with both numeric and non-numeric columns
cancerData_imputed = pd.concat([cancerData[non_numeric_cols], cancerData_numeric_imputed], axis=1)

# Display the processed data
print(cancerData_imputed.head())

# Count missing values after imputation
print("Missing values per column after imputation:")
print(cancerData_imputed.isna().sum())


#encoding