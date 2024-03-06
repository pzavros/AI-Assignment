import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
#data visualisation
import numpy as np
import matplotlib.pyplot as plt







# Load Data
cancerData = pd.read_csv("datasets/Cancer_Data.csv")
breastData = pd.read_csv("datasets/breast-cancer-dataset.csv")
patientData = pd.read_csv("datasets/cancer patient data sets.csv")


print(cancerData)


plt.figure(figsize=(30, 20)) 
# Adjust the layout to prevent overlapping
cancerData.hist(bins=15, layout=(8, 4), figsize=(30, 20))
plt.tight_layout()


plt.show()

# histogram

dep_var = cancerData.iloc[:,:].values 


print(ind_vars)

#preproccesing
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #replace any missing values by the average of the data in the respective column. 
imputer.fit(ind_vars[:, 3:12]) #fit method which computes the mean of the columns specified
ind_vars[:, 3:12] = imputer.transform(ind_vars[:, 3:12]) #the new columns will replace the old columns
print(ind_vars)

