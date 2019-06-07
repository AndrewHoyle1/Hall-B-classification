import h5py
import numpy as np
import sklearn as sk
import tensorflow as tf

f = h5py.File('Train_set', 'r')#opens our training set file to be read

d1 = f.get('neg_train')#unpacks negative training set
n_labels = np.zeros(len(d1))#makes an array of labels for the negative data
d2 = f.get('pos_train')#unpacks positive training set
p_labels = np.ones(len(d2))#makes an array of labels for the positive data

all_data = np.concatenate((d1,d2))#combines negative and positive arrays into one
all_labels = np.concatenate((n_labels,p_labels))#combines label arrays into one

all_data, all_labels = sk.utils.shuffle(all_data, all_labels, random_state = 0)#shuffles both arrays in parallel

all_data = np.float32(all_data)
all_labels = np.float32(all_labels)

d_set = tf.data.Dataset.from_tensor_slices(all_data)
l_set = tf.data.Dataset.from_tensor_slices(all_labels)
batched_set = d_set.batch(32)
batched_labels = l_set.batch(32)
Img_shape = (112,112,3)

for data, label in zip(batched_set.take(1),batched_labels.take(1)):
    pass

base_model = tf.keras.applications.Xception(input_shape = Img_shape, include_top = False, weights = 'imagenet')
feature_batch = base_model(data)

base_model.trainable = False

global_aveerage_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_aveerage_layer(feature_batch)

f1 = h5py.File('data_vectors')
dset_d = f1.create_dataset('vectors', data = feature_batch_average)
dset_l = f1.create_dataset('labels', data = label)
