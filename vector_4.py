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
#import click

"""models to experiment...

    Resnet
    Xception (done)
    InceptionV3
    VGG19
    ResNet50
    MobileNetV2 (done)
    vgg16  (Done)



"""


#@click.command()

#@click.option('--initial_epochs',default=1,help='Number of iterations')
#@click.option('--model_word',default='./Xception_cnn.h5',help="This is the file name that saves the model")


def main ():
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
    ts_d = tf.data.Dataset.from_tensor_slices((t_d,t_l)).shuffle(buffer_size = 1000).batch(batch_size)#training data dataset
    vs_d = tf.data.Dataset.from_tensor_slices((v_d,v_l)).batch(batch_size)#validation data dataset


    
    IMG_SHAPE = (112,112,3)

    base_model = tf.keras.applications.vgg19.VGG19(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')#establish base model
    base_model.trainable = True #freeze model
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1,activation="sigmoid")
    model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])#add new layers onto base_model
    base_learning_rate = 2e-5#base learning rate
    optimizer = tf.keras.optimizers.Nadam(lr = base_learning_rate) #define our optimizer to be ADAM and set our learnitn rate


    initial_epochs =7  #the number of iterations 


    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])#compiles model





    checkpoint_path= "./vgg19_checkpoints/cp-{epoch:04d}.ckpt"  #Defines where to create the file to put our checkpoints in.
    #checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, # Tells our callback where to put each checkpoint per iteration/epoch
                                                    save_weights_only=True, #this saves our weights from each iteration/epoch
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode="auto")#Sets the verbosity

    #tf.keras.callbacks.EarlyStopping(monitor=v_d,patience=5,mode="max",)



    history = model.fit(ts_d, epochs = initial_epochs,callbacks = [cp_callback],validation_data = vs_d)#trains model dependent on the epochs wanted
    model_word= './vgg19_cnn.h5'

    model.save(model_word)
    
    
    new_model = keras.models.load_model(model_word)
    prediction=new_model.predict(t_d)
    prediction=np.round(prediction)
    print (prediction.shape)
    cm= sklearn.metrics.confusion_matrix(t_l,prediction)  #had v_d before
    print(cm)

    print ('Accuracy Score :',sklearn.metrics.accuracy_score(t_l, prediction) )
    print ('Report : ')
    print (sklearn.metrics.classification_report(t_l, prediction)) 

    df_cm=pd.DataFrame(cm)
    sn.set(font_scale=1.4)
    sns_plot =sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, fmt='g')# font size
    figure=sns_plot.get_figure()
    figure.savefig("cm_vgg19.png",dpi=400)
    
    

    



    
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
    plt.savefig('vgg19_loss_accuracy_graph')
    


if __name__ == '__main__':
    main()