import h5py
import numpy as np
from sklearn import model_selection
import sklearn
import tensorflow as tf
import PIL
import glob
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

f = h5py.File('Train_set', 'r')#opens our training set file to be read

d1 = f.get('neg_train')#unpacks negative training set
n_labels = np.zeros(len(d1))#makes an array of labels for the negative data
d2 = f.get('pos_train')#unpacks positive training set
p_labels = np.ones(len(d2))#makes an array of labels for the positive data

all_data = np.concatenate((d1,d2))#combines negative and positive arrays into one
all_labels = np.concatenate((n_labels,p_labels))#combines label arrays into one

all_data, all_labels = sklearn.utils.shuffle(all_data, all_labels, random_state = 0)#shuffles both arrays in parallel
all_data = np.float32(all_data)#changes dtype to preferred float32
all_labels = np.float32(all_labels)
 

t_d, v_d, t_l, v_l = sklearn.model_selection.train_test_split(all_data, all_labels, test_size = 0.25)
batch_size = 32
t_d = tf.data.Dataset.from_tensor_slices((t_d,t_l)).shuffle(buffer_size = 1000).batch(batch_size)
v_d = tf.data.Dataset.from_tensor_slices((v_d,v_l)).batch(batch_size)
base_learning_rate = 0.00008

initial_epochs = 10
steps_per_epoch = 32
validation_steps = 20
IMG_SHAPE = (112,112,3)

model=tf.keras.models.Sequential()
model.add(Conv2D(64, (3, 3), input_shape=IMG_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(t_d,epochs = initial_epochs, validation_data=v_d)  #batch_size=32


"""
model = tf.keras.models.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              lr = base_learning_rate)

history = model.fit(t_d, epochs = initial_epochs, validation_data = v_d)
"""