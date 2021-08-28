import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import FunctionTransformer

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

# minMaxScaler - Transform features by scaling each feature to a given range
# The transformation is given by:
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
# where min, max = feature_range.
# by default, scales data between 0 and 1
min_max_scaler = MinMaxScaler((2,3)).fit(arr)
print(min_max_scaler.scale_)    #Per feature relative scaling of the data
print('Column wise min',min_max_scaler.data_min_) #Per feature minimum seen in the actual data
print('Column wise max',min_max_scaler.data_max_) #Per feature maximum seen in the actual data
min_max_scaled = min_max_scaler.transform(arr)
print(min_max_scaled)
#(or) use fit_trasnform
min_max_scaled = MinMaxScaler((2,3)).fit_transform(arr)
print(min_max_scaled)

# MaxAbsScaler() - Scale each feature by its maximum absolute value.
# The MaxAbsScaler works very similarly to the MinMaxScaler but automatically scales the data to a [-1,1] 
# range based on the absolute maximum. 
# ie., x_scaled = x / max(abs(x))
# max(abs(x)) is calculated **column wise** (axis = 0)
max_abs_scaler = MaxAbsScaler().fit(arr)
max_abs_scaled = max_abs_scaler.transform(arr)
print('max_abs_scaled\n',max_abs_scaled)
# column-wise max if find out, and each element is divided by corresponding column-wise max.

# FunctionTransformer - A FunctionTransformer forwards its data to a user-defined function or function object 
# and returns the result of this function.

# FunctionTransformer - using user-defined function
def add_one(x):
	return x+1
scaler = FunctionTransformer(add_one)
scaled = scaler.fit_transform(arr)  #apply the add_one function to all input data elements
print(scaled)

# FunctionTransformer - using function object
scaler = FunctionTransformer(np.log1p)  # np.log1p == np.log(1+input_data)
scaled = scaler.fit_transform(arr)      #apply the np.log1p function to all input data elements
print(scaled)
