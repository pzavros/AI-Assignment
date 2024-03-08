import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

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
print(cancerData_imputed[['diagnosis', 'diagnosis_encoded']].head(100))

# Count missing values after imputation
print("Missing values per column after imputation:")
print(cancerData_imputed.isna().sum())
