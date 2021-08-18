#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
import numpy as np


# In[29]:


"""
@author Christian Gebhardt
@email christian.gebhardt@uni-bayreuth.de
"""
# Script that shows how to generate cross validation indices for own classifiers and .npz file for comparison classifiers
# and train/test own classifier with them

# Method to generate k x n cross validation trainings- and test indices.
def generate_kxn_cv(X, y, k, n):
    train_indices = []
    test_indices = []
    
    assert n > 0, "n must be greater than 0"
    assert k > 0, "k must be greater than 0"
        
    for i in range(k):
        skf = StratifiedKFold(n_splits=n, shuffle=True)
        skf.get_n_splits(X, y)
        train_index, test_index = skf.split(X, y)
        for j in range(n):
            train_indices.append(train_index[j])
            test_indices.append(test_index[j])
        
    return (train_indices, test_indices)

# Example dataset, replace this with your feature data matrix X and target vector y.
df = load_iris()
X = df.data
y = df.target

# Example indices with k=5 and n=2, replace this with your desired k and n.
k = 5
n = 2
train_indices, test_indices = generate_kxn_cv(X, y, k, n)

# print("train_indices before saving:\n{0}".format(train_indices)) # uncomment this line to check train indices
# print("test_indices before saving:\n{0}".format(test_indices)) # uncomment this line to check test indices

# Saving train_indices and test_indices to a .npz file for input in evaluation framework. You can use them now to train and test your classifier, 
# remember to use this file as later input for comparison classifiers. You can replace file with your desired filename (or output stream)
# but the keys 'train_indices' and 'test_indices' should stay the same.

np.savez(file="train_test_indices", train_indices=train_indices, test_indices=test_indices)

# Loading train_indices and test_indcies for possible later use.

indices = np.load(file="train_test_indices.npz")

# Access with keys 'train_indices' and 'test_indices'.

train_indices = indices['train_indices']
test_indices = indices['test_indices']

# print("train_indices after loading:\n{0}".format(train_indices)) # uncomment this line to check train indices
# print("test_indices after loading:\n{0}".format(test_indices)) # uncomment this line to check test indices

# Use indices like this for training and testing with own classifier (example with sklearn classifier)
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000)

y_pred = []
for i in range(len(train_indices)):
    clf.fit(X[train_indices[i]], y[train_indices[i]])
    y_pred.append(clf.predict(X[test_indices[i]]))

y_pred = np.asarray(y_pred)

# print("Results CV before saving:\n{0}".format(y_pred))  # uncomment this line to check results y_pred

# Save them as .npy file for later input in evaluation framework, replace with desired filename.
np.save("y_pred_own_cv", y_pred)

# Loading npy file for possible later use.
y_pred = np.load("y_pred_own_cv.npy")

# print("Results CV after loading:\n{0}".format(y_pred))  # uncomment this line to check results y_pred


# In[ ]:




