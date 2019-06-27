import h5py
import numpy as np
from sklearn import model_selection
import sklearn
import tensorflow as tf
import PIL
import glob
from matplotlib import pyplot as plt
import os

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
t_d, v_d, t_l, v_l = sklearn.model_selection.train_test_split(all_data, all_labels, test_size = 0.25)#splits data and labels into training and validation sets
batch_size = 32
t_d = tf.data.Dataset.from_tensor_slices((t_d,t_l)).shuffle(buffer_size = 1000).batch(batch_size)#training data dataset
v_d = tf.data.Dataset.from_tensor_slices((v_d,v_l)).batch(batch_size)#validation data dataset


#def create_model():

IMG_SHAPE = (112,112,3)

base_model = tf.keras.applications.vgg16.VGG16(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')#establish base model
base_model.trainable = True #freeze model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])#add new layers onto base_model
base_learning_rate = 2e-6#base learning rate
optimizer = tf.keras.optimizers.Adam(lr = base_learning_rate) #define our optimizer to be ADAM and set our learnitn rate


initial_epochs = 25  #the number of iterations 


model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])#compiles model





checkpoint_path= "./Vgg16_checkpoints/cp-{epoch:04d}.ckpt"  #Defines where to create the file to put our checkpoints in.
#checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)           #
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, # Tells our callback where to put each checkpoint per iteration/epoch
                                                 save_weights_only=True, #this saves our weights from each iteration/epoch
                                                 verbose=1, #Sets the verbosity
                                                 period=1) #This tells how many epochs


#checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model) 

#status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

#checkpoint.save(file_prefix=checkpoint_prefix)


history = model.fit(t_d, epochs = initial_epochs,callbacks = [cp_callback],validation_data = v_d)#trains model dependent on the epochs wanted




acc = history.history['accuracy']#plots accuracy and loss over each epoch
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(acc, label = 'Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2,1,2)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,7.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('loss_accuracy_graph')
