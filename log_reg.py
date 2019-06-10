import h5py
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
import numpy as np
import matplotlib
import pylab as plt

f = h5py.File('data_vectors', 'r')#opens file

dset_d = f.get('vectors')#gets vector data
dset_l = f.get('labels')#gets label data

dataset = []#empty list for data
labels = []#empty list for labels

for data, label in zip(dset_d, dset_l):#unloads data into lists
    dataset.append(data)
    labels.append(label)

training_d, validation_d, training_l, validation_l = sklearn.model_selection.train_test_split(dataset, labels, test_size = 0.25)#splits data into training and validation sets

logisticRegr = LogisticRegression()#calls logistic regression algorithm from scikit learn

logisticRegr.fit(training_d,training_l)#trains with the training set
predictions = logisticRegr.predict(validation_d)#makes predictions using the validation sets
score = logisticRegr.score(validation_d,validation_l)#assesses accuracy
print(score)

cm = metrics.confusion_matrix(validation_l, predictions)#creates confusion matrix from our results

print(cm)
