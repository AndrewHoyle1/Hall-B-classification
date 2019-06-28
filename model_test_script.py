#This is a validation scrit that will use the checkpoints to make a prediction from the dataset and see if the differences.

#step 1 load the checkpoint from the file
#Load the test data for the model
#Run the dataset through the model to see how much accracy it has.
import os
import tensorflow as tf
from tensorflow import keras
import h5py
import numpy as np
from sklearn import model_selection
import sklearn
import PIL
import glob
from matplotlib import pyplot as plt


f = h5py.File('Test_set', 'r')#opens our training set file to be read

d1 = f.get('neg_test')#unpacks negative training set
n_labels = np.zeros(len(d1))#makes an array of labels for the negative data, this means our "zeros" represents negative
d2 = f.get('pos_test')#unpacks positive training set
p_labels = np.ones(len(d2))#makes an array of labels for the positive data

all_data = np.concatenate((d1,d2))#combines negative and positive arrays into one
all_labels = np.concatenate((n_labels,p_labels))#combines label arrays into one

all_data, all_labels = sklearn.utils.shuffle(all_data, all_labels, random_state = 0)#shuffles both arrays in parallel
all_data = np.float32(all_data)#changes dtype to preferred float32
all_labels = np.float32(all_labels)
#testing = tf.data.Dataset.from_tensor_slices((all_data,all_labels)).shuffle(buffer_size=1000).batch(batch_size=1)




def create_model():


    IMG_SHAPE = (112,112,3)


    base_model = tf.keras.applications.vgg16.VGG16(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')#establish base model
    base_model.trainable = True #freeze model
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)
    model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])#add new layers onto base_model
    base_learning_rate = 2e-6#base learning rate
    optimizer = tf.keras.optimizers.Adam(lr = base_learning_rate) #define our optimizer to be ADAM and set our learnitn rate


    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])#compiles model
    
    return model


new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(all_data, all_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))   



checkpoint_path= "./Vgg16_checkpoints/cp-{epoch:04d}.ckpt"  #Defines where to create the file to put our checkpoints in.

checkpoint_dir = os.path.dirname(checkpoint_path)


latest = tf.train.latest_checkpoint(checkpoint_dir)
print (latest)

#checkpoint.restore(latest)

model=create_model()
model.load_weights(latest)
#model.load_weights(checkpoint_path)


loss,acc = model.evaluate(all_data, all_labels)
    
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#for img,label in testing:
    

    #loss.append=[loss]
    #acc.append[acc]
    




#model.load_weights(checkpoint_path)
#model.load_weights(latest)  
