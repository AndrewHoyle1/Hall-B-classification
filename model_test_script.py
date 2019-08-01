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
import pandas as pd
import seaborn as sn


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


def create_model():


    IMG_SHAPE = (112,112,3) # size of the image array


    base_model = tf.keras.applications.vgg19.VGG19(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')#establish base model
    base_model.trainable = True  #this sets our training to either true or false
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #this is a global pooling layer
    prediction_layer = tf.keras.layers.Dense(1,activation="sigmoid")    #this is a dense layer with a sigmoid activation
    model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])#add new layers onto base_model
    base_learning_rate = 2e-5 #base learning rate
    optimizer =  tf.keras.optimizers.Nadam(lr = base_learning_rate) #define our optimizer to be NaDAM and set our learnitn rate


    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])#compiles model
    
    return model    #Returns our partial formed model

model='./vgg19_cnn.h5'      # the location of the model

new_model = keras.models.load_model(model)  #loads our model from the location already defined in new model.
new_model.summary()                         #gives a summary of the parameters in the model
loss, acc = new_model.evaluate(all_data, all_labels)    #Evaluates the accuracy using the test data and labels

new_model = keras.models.load_model(model) #loads our model
prediction=new_model.predict(all_data)  #make our model make prediction from our test data
prediction=np.round(prediction)         #rounds our models predictions to 1 or 0
print (prediction.shape)                #print the size of our prediction array
cm= sklearn.metrics.confusion_matrix(all_labels,prediction)  #Preforms our confusion matrix calculations
print(cm)   #prints our confucion matrix

print ('Accuracy Score :',sklearn.metrics.accuracy_score(all_labels, prediction) )  #give the accracy of our confusion matrix
print ('Report : ')         #prints report
class_report=sklearn.metrics.classification_report(all_labels, prediction) #Crates a classication report from our confusion matrix
print(class_report) #Prints our classication report

df_cm=pd.DataFrame(cm)                                                  #Creates a Pandas database out of the prediction
sn.set(font_scale=1.4)                                                  #sets the font size of the seaborn were using to print the confusion matrix
sns_plot =sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, fmt='g') #this allows use to define how our confusion matrix will look
figure=sns_plot.get_figure()                                            #allows to create the heat map of the confusion matrix
figure.savefig("cm_test_vgg19_test_2.png",dpi=400)                      #this save the confusion matrix


file = open('./vgg19_test_class_report_1.2.txt', 'w') #makes a file for our classication report
file.write(class_report)                              #stores our classication report inside a txt file
file.close()                                          #close the creation of the txt file





checkpoint_path= "./vgg19_checkpoints/cp-{epoch:04d}.ckpt"          #Defines where to create the file to put our checkpoints in.

checkpoint_dir = os.path.dirname(checkpoint_path)                   #directs where our file will be


latest = tf.train.latest_checkpoint(checkpoint_dir)                 #gets the latest checkpoint made
#print (latest)                                                     #prints the name of the file

#checkpoint.restore(latest)                                         #restores lastest checkpoint

model=create_model()                                                #Puts our model in this script into a variable
model.load_weights(latest)                                          #loads the weights from our checkpoint


loss,acc = model.evaluate(all_data, all_labels)                     #Judges the accuracy of the checkpoint made
    
print("Restored Checkpoint, accuracy: {:5.2f}%".format(100*acc))    # Prints our accuracy