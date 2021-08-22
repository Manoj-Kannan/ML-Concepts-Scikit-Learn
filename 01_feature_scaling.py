import numpy as np
from sklearn.preprocessing import StandardScaler

arr = np.random.randint(low=1,high=10,size=(3,3))
print('Data:',arr)

# Standard Scaler - converts the features into zero mean and unit variance
# The standard score of a sample x is calculated as: z = (x - u) / s
# where u is the mean of the training samples or zero if with_mean=False, 
# and s is the standard deviation of the training samples
# by default, scales data between -1 and 1
std_scaler = StandardScaler().fit(arr)
print(std_scaler.mean_)     #Per feature relative scaling of the data to achieve zero mean and unit variance
print(std_scaler.scale_)    #The mean value for each feature in the training set
std_scaled = std_scaler.transform(arr)
print(std_scaled)
#after scaling mean = 0, std = 1
print(std_scaled.mean(axis=0))
print(std_scaled.std(axis=0))
#(or) use fit_trasnform
std_scaled = StandardScaler().fit_transform(arr)
print(std_scaled)
