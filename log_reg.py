import h5py
from sklearn.linear_model import LogisticRegression
from sklearn import metrics,manifold
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from matplotlib import pyplot as plt

f = h5py.File('data_vectors', 'r')#opens file

dset_d = f.get('vectors')#gets vector data
dset_l = f.get('labels')#gets label data

dataset = []#empty list for data
labels = []#empty list for labels

for data, label in zip(dset_d, dset_l):#unloads data into lists
    dataset.append(data)
    labels.append(label)

tsne = manifold.TSNE(n_components = 2)

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
data_min_max = min_max_scaler.fit(dataset)
data_min_max = min_max_scaler.transform(dataset)

"""data_2d = tsne.fit_transform(data_min_max[:1000])

data_one = data_2d[:,0]
data_two = data_2d[:,1]

plt.figure(figsize = (3,3))
sns.scatterplot(x = data_one, y = data_two, hue = labels[:1000], palette = sns.color_palette("hls", 2))
plt.show()"""


training_d, validation_d, training_l, validation_l = sklearn.model_selection.train_test_split(data_min_max, labels, test_size = 0.25)#splits data into training and validation sets

#rf = LogisticRegression()#calls logistic regression algorithm from scikit learn
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(training_d,training_l)#trains with the training set
predictions = rf.predict(training_d)#makes predictions using the validation sets
predict2 = rf.predict(validation_d)
score = rf.score(training_d,training_l)#assesses accuracy
score2 = rf.score(validation_d,validation_l)
print(score,score2)
