# RobustScaler - Robust Scaler algorithms scale features that are robust to outliers. 
# The method it follows is almost similar to the MinMax Scaler but it uses the interquartile range(IQR). 
# It transforms the feature vector by subtracting the median(50%) and then dividing by the interquartile range (75% value — 25% value).
# ie., value = (value – median) / (Q75 – Q25)
# where, Median(50%) - value at position 25% of entire input data
# Similarly, 25%,75% - values at position 25% and 75% of entire input data	
# Q25, Q50, Q75 are calculated column wise (axis = 0)

import numpy as np
from sklearn.preprocessing import RobustScaler

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("array\n",arr)

scaler = RobustScaler().fit(arr)
scaled = scaler.transform(arr)
print("scaled\n",scaled)

# Explanation
# To determine Q25, Q50, Q75
# Q = np.quantile(arr,[0.25,0.5,0.75],axis=0)
#calculate individually
Q25 = np.quantile(arr,0.25,axis=0)
Q50 = np.quantile(arr,0.50,axis=0)
Q75 = np.quantile(arr,0.75,axis=0)
print("quantiles\n",Q25)
print(Q50)
print(Q75)

#scaled_value = (value – Q50) / (Q75 – Q25)
scaled_00 = (arr[0][0]-Q50[0])/(Q75[0]-Q25[0])
scaled_11 = (arr[1][1]-Q50[1])/(Q75[1]-Q25[1])
scaled_22 = (arr[2][2]-Q50[2])/(Q75[2]-Q25[2])
print(scaled_00)
print(scaled_11)
print(scaled_22)

# Also Refer: https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/