import h5py
import numpy as np
from sklearn import model_selection
import sklearn
import tensorflow as tf
import PIL
import glob
from matplotlib import pyplot as plt
from tensorflow import keras

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

IMG_SHAPE = (112,112,3)

base_model = tf.keras.applications.vgg16.VGG16(input_shape = IMG_SHAPE, include_top = False, weights = None)
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l2(0.005),activation="relu",use_bias=True)
drop_layer=keras.layers.Dropout(0.05)
#batch_normal=tf.keras.layers.BatchNormalization()
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer,drop_layer]) #,batch_normal
base_learning_rate = 0.00008
model.compile(optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999,lr = base_learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'],)

#print(base_model.summary())

initial_epochs = 10
steps_per_epoch = 32
validation_steps = 20

history = model.fit(t_d, epochs = initial_epochs, validation_data = v_d)

results = model.evaluate(v_d)