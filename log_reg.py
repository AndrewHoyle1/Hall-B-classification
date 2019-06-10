import h5py
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
import numpy as np
import matplotlib
import pylab as plt

f = h5py.File('data_vectors', 'r')

dset_d = f.get('vectors')
dset_l = f.get('labels')

dataset = []
labels = []

for data, label in zip(dset_d, dset_l):
    dataset.append(data)
    labels.append(label)

training_d, validation_d, training_l, validation_l = sklearn.model_selection.train_test_split(dataset, labels, test_size = 0.25)

logisticRegr = LogisticRegression()

logisticRegr.fit(training_d,training_l)
predictions = logisticRegr.predict(validation_d)
score = logisticRegr.score(validation_d,validation_l)
print(score)

cm = metrics.confusion_matrix(validation_l, predictions)

print(cm)
