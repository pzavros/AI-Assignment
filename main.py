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
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #replace any missing values by the average of the data in the respective column. 
imputer.fit(dep_var[:,:]) #fit method which computes the mean of the columns specified
dep_var[:,:] = imputer.transform(dep_var[:,:]) #the new columns will replace the old columns
print(dep_var)



