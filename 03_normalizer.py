import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize # used to get norm value

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('array\n',arr)

# Normalizer - Normalization is the process of scaling individual samples to have unit norm.
# When to normalize? Normalize data when the algorithm predicts based on the weighted relationships formed between data points.
# Important Note:
# One of the key differences between scaling (e.g. standardizing) and normalizing, is that 
# normalizing is a **row-wise** operation(axis=1), while scaling is a **column-wise** operation(axis=0).
# Data can be normalized in 3 ways: max, l1, l2(default).
# max: x_normalized = x / max(x) 	ie. row-wise max
# l1 : x_normalized = x / sum(X) 	ie. row-wise sum
# l2 : x_normalized = x / sqrt(sum((i**2) for i in X))

#norm='max'
max_normalizer = Normalizer(norm='max').fit(arr)
max_normalized = max_normalizer.transform(arr)
#norm='l1'
l1_normalizer  = Normalizer(norm='l1').fit(arr)
l1_normalized  = l1_normalizer.transform(arr)
#norm='l2'
l2_normalizer  = Normalizer(norm='l2').fit(arr)
l2_normalized  = l2_normalizer.transform(arr)

print('max_normalized\n',max_normalized)
print('l1_normalized\n',l1_normalized)
print('l2_normalized\n',l2_normalized)
