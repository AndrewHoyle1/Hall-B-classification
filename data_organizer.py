import h5py
import numpy as np

f = h5py.File('Num_data', 'r')#calls file made in previous script

neg_data = []#empty list for our negative data
pos_data = []#empty list for our positive data

d_neg = f.get('neg')#imports negative dataset from file
d_pos = f.get('pos')#imports negative dataset from file

for data in d_neg:#unpacks negative data in dataset and sends it to neg_data list
    neg_data.append(data)
for data in d_pos:#unpacks positive data and sends it to list
    pos_data.append(data)

np.random.shuffle(neg_data)#shuffles negative data
np.random.shuffle(pos_data)#shuffles positive data

neg_test = []#empty list for negative test set
neg_train = []#empty list for negative training set
pos_test = []#empty list for positive test set
pos_train = []#empty list for positive training set

for i in range(len(neg_data)):#itterates through the indexes of the neg_data list
    if i <= len(neg_data)*7.5//10:#sorts into training and testing lists based on which index it is
        neg_train.append(neg_data[i])#the if statement is written to have a roughly 80/20 split between the size of the sets
    elif i > len(neg_data)*7.5//10:
        neg_test.append(neg_data[i])
for i in range(len(pos_data)):#performs the same operations as the above loop for the positive data
    if i <= len(pos_data)*7.5//10:
        pos_train.append(pos_data[i])
    elif i > len(pos_data)*7.5//10:
        pos_test.append(pos_data[i])

f1 = h5py.File('Train_set')#makes a file for the training set
f2 = h5py.File('Test_set')#makes a file for the testing set

dset_n1 = f1.create_dataset('neg_train', data = neg_train)#sends neg_train list to proper file
dset_n2 = f2.create_dataset('neg_test', data = neg_test)#sends neg_test list to proper file
dset_p1 = f1.create_dataset('pos_train', data = pos_train)#sends pos_train list to proper file
dset_p2 = f2.create_dataset('pos_test', data = pos_test)#sends pos_test list to proper file
