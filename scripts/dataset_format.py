#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


# In[22]:


"""
@author Christian Gebhardt
@email christian.gebhardt@uni-bayreuth.de
"""
# Script that shows how to bring datasets in a supported format.

# If you have data (X, y) already just use .npy format

df = load_iris()
X = df.data
y = df.target

print("Shapes of X and y before reshaping: {0}, {1}".format(X.shape, y.shape))

# bring data in supported shapes and stack X (nxm) and y (nx1) together in one matrix (nxm+1), with last column (!) as target column
y = y.reshape((150, 1))

dataset = np.hstack((X, y))
print("Final dataset:\n {0}".format(dataset))

# saving dataset as .npy (replace filename as desired)
np.save("dataset_iris", dataset)

# If you have a csv just send it as csv (first row must be header, target variable in last column, seperator must be ','), using a example from Github here
# you can pass any url with csv content here

dataset = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
# dataset.head() # uncomment this line to see dataset

# saving file as csv (make sure to use .csv suffix)
dataset.to_csv("dataset_diabetes.csv", sep=',', encoding='utf-8')

# If you are using classification datasets from the machine learning data repository (or any other repositoy) at https://archive.ics.uci.edu/ml/index.php
# or in general .data files you can also read them with pandas (but you need to know the seperator), often they don't have a header,
# so make sure to use header=None

dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", sep=',', header=None)
# dataset.head() # uncomment this line to see dataset

# converting csv file to numpy (since this dataset has no header line, alternativly you can add a fake header row)
dataset = dataset.to_numpy()

# saving dataset as .npy (target column is already last column, if not change format)
np.save("dataset_wine", dataset)

# Finally if the dataset contains categorical variables as strings or missing values, you need to convert them in a sklearn supported format
# see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder,
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
# and https://scikit-learn.org/stable/modules/impute.html for more information


# In[ ]:




