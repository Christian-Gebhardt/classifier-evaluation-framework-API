#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import numpy as np


# In[18]:


"""
@author Christian Gebhardt
@email christian.gebhardt@uni-bayreuth.de
"""
# Script that shows how to get y_true (true label vector), y_pred (predicted label vector) or y_proba (probability matrix)
# as input for evaluation framework

# Example dataset, replace this with your feature data matrix X and target vector y.
df = load_iris()
X = df.data
y = df.target

# Now split them into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# y_test is already the true label vector for later evaluation, so save y_test in .npy format
np.save("y_true_iris", y_test)

# Train your classifier with training data and predict test labels (this might be a different process for your own model/classifier
# but in the end you should have a vector with predicted labels)

clf = SGDClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Save y_pred in .npy format
np.save("y_pred_iris", y_test)

# Finally if you want to analyze probabilistic values from the prediction of your classifier (or any sklearn classifier)
# you need to have a nxc matrix, where n is the number of instances and c is the number of classes. So in every row there should
# be all probabilities for each class. With most sklearn classifiers you can do the following:

# This works for the SGD classifier we defined previously
calibrator = CalibratedClassifierCV(clf, cv='prefit')
model = calibrator.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)

# print("Probabilistic matrix:\n{0}".format(y_proba_iris)) # uncomment this line to check probabilistic matrix format

# save y_proba in .npy format
np.save("y_proba_iris", y_proba)


# In[ ]:




