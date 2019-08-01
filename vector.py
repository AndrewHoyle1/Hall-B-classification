import h5py
import numpy as np
from sklearn import model_selection
import sklearn
import tensorflow as tf
import PIL
import glob
from matplotlib import pyplot as plt
import os
from tensorflow import keras
import pandas as pd
import seaborn as sn

f = h5py.File('Train_set', 'r')#opens our training set file to be read

d1 = f.get('neg_train')#unpacks negative training set
n_labels = np.zeros(len(d1))#makes an array of labels for the negative data
d2 = f.get('pos_train')#unpacks positive training set
p_labels = np.ones(len(d2))#makes an array of labels for the positive data

all_data = np.concatenate((d1,d2))#combines negative and positive arrays into one
all_labels = np.concatenate((n_labels,p_labels))#combines label arrays into one

all_data, all_labels = sklearn.utils.shuffle(all_data, all_labels, random_state = 0)#shuffles both arrays in parallel
all_data = np.float32(all_data)#changes dtype to preferred float32
all_labels = np.float32(all_labels) #changes dtype to preferred float32
t_d, v_d, t_l, v_l = sklearn.model_selection.train_test_split(all_data, all_labels, test_size = 0.25) #splits data and labels into training and validation sets
batch_size = 32
t_d = tf.data.Dataset.from_tensor_slices((t_d,t_l)).shuffle(buffer_size = 1000).batch(batch_size) #create dataset from training labels and data
v_d = tf.data.Dataset.from_tensor_slices((v_d,v_l)).batch(batch_size)    #create dataset from validation labels and data

IMG_SHAPE = (112,112,3)     # size of the image array

base_model = tf.keras.applications.vgg16.VGG16(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet') #establish base model
base_model.trainable = True     #this sets our training to either true or false
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()     #this is a global pooling layer
prediction_layer = tf.keras.layers.Dense(1,activation="sigmoid")    #this is a dense layer with a sigmoid activation
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])   #compiles parameters
base_learning_rate =2e-5    #Learning rate
model.compile(optimizer = tf.keras.optimizers.Nadam(lr = base_learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])# compiles parameters

#print(base_model.summary())

initial_epochs = 10 #this is the number of epochs we want for training

checkpoint_path= "./vgg16_checkpoints/cp-{epoch:04d}.ckpt"  #Defines where to create the file to put our checkpoints in.
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,        #Tells our callback where to put each checkpoint per iteration/epoch
                                                monitor='v_d',           # Tells the callback to monitor a parameter in the model.
                                                save_weights_only=True,  # this saves our weights from each iteration/epoch
                                                save_best_only=False,    # This will create a checkpoint per epoch if False
                                                mode='auto',             # This sets monitors for increase or decrease from your monitor parameter
                                                verbose=1,)              # This sets the verbosity
                                                    



history = model.fit(t_d, epochs = initial_epochs,callbacks = [cp_callback], validation_data = v_d) #trains model 

model_word= './vgg16_cnn.h5' #this is the location that the file model is being saved

model.save(model_word)       #this saves the model in the location we gave in model_word


new_model = keras.models.load_model(model_word) #loads in the model that was trained
prediction=new_model.predict(t_d)   #uses our models to make a predictions made from our training data
prediction=np.round(prediction)     #Rounds our predictions to 1 or 0
print (prediction.shape)            #prints the shape of the array that has our prediction
cm= sklearn.metrics.confusion_matrix(t_l,prediction)  #this creates the confusion matrix using the training labels and predictions
print(cm) #prints our our confusion matrix

print ('Accuracy Score :',sklearn.metrics.accuracy_score(t_l, prediction) ) #gives a accuracy score for our given prediction
print ('Report : ') #prints report:
print (sklearn.metrics.classification_report(t_l, prediction)) #Creates a classification report in the terminal to see the different metrics

df_cm=pd.DataFrame(cm) #Creates a Pandas database out of the prediction
sn.set(font_scale=1.4)  #sets the font size of the seaborn were using to print the confusion matrix
sns_plot =sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, fmt='g')# this allows use to define how our confusion matrix will look
figure=sns_plot.get_figure() #allows to create the heat map of the confusion matrix
figure.savefig("cm_vgg16.png",dpi=400)  #this save the confusion matrix




acc = history.history['accuracy']                   #create a list of variable from our percent in accuracy
val_acc = history.history['val_accuracy']           #create a list of variable from our percent in validation accuracy

loss = history.history['loss']                      #create a list of variable from our percent in loss
val_loss = history.history['val_loss']              #create a list of variable from our percent in validation loss

plt.figure(figsize = (10,10))                       #size of the figure
plt.subplot(2,1,1)
plt.plot(acc, label = 'Training Accuracy')  #create a label for training accuracy
plt.plot(val_acc, label = 'Validation Accuracy')    #create a label for validation accuracy
plt.legend(loc = 'lower right')                     #The place where our labels will go
plt.ylabel('Accuracy')                              #Labels the y axis
plt.ylim([0.6,1])                                   #The range for the y axis
plt.title('Training and Validation Accuracy')       #The title for the first graph

plt.subplot(2,1,2)
plt.plot(loss, label = 'Training Loss')             #monitors our training loss varibles
plt.plot(val_loss, label = 'Validation Loss')       #monitors our training validation loss
plt.legend(loc = 'lower right')                     #Gives the locations for our labels
plt.ylabel('Cross Entropy')                         #The label for our y axis
plt.ylim([-0.25,0.5])                               #The range of the y axis
plt.title('Training and Validation Loss')           #Give the title of our second
plt.xlabel('epoch')                                 #Our x axis  monitored 
plt.savefig('loss_accuracy_graph')                  #Saves our graphs

