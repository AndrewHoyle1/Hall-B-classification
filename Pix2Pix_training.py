import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, utils
import os
import time
from IPython.display import clear_output

f = h5py.File('Pix2Pix_data_2', 'r')
Event = f.get('all')
Event = np.array(Event)
Track = f.get('extracted')
Track = np.array(Track)

BUFFER_SIZE = 400
BATCH_SIZE = 32

event, track = utils.shuffle(Event, Track, random_state = 0)#shuffles event and track data in unison
event = np.float32(event)/255#converts event data to float32
track = np.float32(track)/255#converts track data to float32
event_train, event_test, track_train, track_test = model_selection.train_test_split(event, track, test_size = 0.25, shuffle = False)#splits into training and testing sets
event_train, event_val, track_train, track_val = model_selection.train_test_split(event_train, track_train, test_size = 0.25, shuffle = False)

train_dataset = tf.data.Dataset.from_tensor_slices((event_train,track_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)#creates tensorflow training dataset
test_dataset = tf.data.Dataset.from_tensor_slices((event_test, track_test)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)#creates tensorflow testing dataset
val_dataset = tf.data.Dataset.from_tensor_slices((event_val, track_val)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3#number of output channels for our images

def downsample(filters, size, apply_batchnorm = True):#downsamples images by a factor of 2
    initializer = tf.random_normal_initializer(0., 0.02)#instantiates an initializer

    result = tf.keras.Sequential()#establishes sequential model where our new layers will go
    result.add(tf.keras.layers.Conv2D(filters, size, strides = 2, padding = 'same', kernel_initializer= initializer, use_bias = False))#makes convolutional layers with a depth of the number of filters, and a size of half the previous layer
    # the new layers are downsized by a factor of two since our stride is two but our padding is same
    if apply_batchnorm:#applies batch normalization when needed
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())#applies LeakyReLU activiation
    return result

def upsample(filters, size, apply_dropout = False):#upsamples images by a factor of 2
    initializer = tf.random_normal_initializer(0., 0.02)#initializer

    result = tf.keras.Sequential()#sequential model for new layers
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides = 2, padding = 'same', kernel_initializer = initializer, use_bias = False))#convolutional transpose layer where we upsize

    result.add(tf.keras.layers.BatchNormalization())#batch normalization

    if apply_dropout:#applies dropout when instructed (dropout essentially selects random neurons to ignore when training)
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())#applies ReLU

    return result

def Generator():#generates our images from our input images
    down_stack = [
        downsample(64,4,apply_batchnorm=False),# layers of size (bs, 56, 56, 64)
        downsample(128,4),# (bs, 28,28, 128)
        downsample(256,4),# (bs, 14, 14, 256)
        downsample(512,4),# (bs, 7, 7, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout = True),# (bs, 14, 14, 1024)
        upsample(256,4),# (bs, 28, 28, 512)
        upsample(128, 4),# (bs, 56, 56, 256)
    ]

    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides = 2, padding = 'same', kernel_initializer = initializer, activation = 'tanh')# (bs, 112, 112, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape = [None, None, 3])
    x = inputs
    # downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    # upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x,skip])

    x = last(x)

    return tf.keras.Model(inputs = inputs, outputs = x)

generator = Generator()

def Discriminator():#analyzes our images and evaluates how good they are
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape = [None, None, 3], name = 'input_image')#where we will assign our input image
    tar  = tf. keras.layers.Input(shape = [None, None, 3], name = 'target_image')#where we will assign our target image

    x = tf.keras.layers.concatenate([inp, tar])#concatenates our images (bs, 112,112,channels*2)

    down1 = downsample(64,4,False)(x)#downsamples (bs, 56, 56, 64)
    down2 = downsample(128, 4)(down1)#downsamples (bs, 28, 28, 128)
    down3 = downsample(256, 4)(down2)#downsamples (bs, 14, 14, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(x)#zero pads our most recent layer (bs, 16, 16, 256)
    conv = tf.keras.layers.Conv2D(512,4,strides = 1, kernel_initializer = initializer, use_bias = False)(zero_pad1)#(bs, 12, 12, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)# does batchnormalization on our most recent layer

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)#applies a leaky ReLU function to our batch normalized layer

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)#zero pads our last layer (bs, 13, 13, 512)

    last = tf.keras.layers.Conv2D(1,4, strides = 1, kernel_initializer= initializer)(zero_pad2)#(bs, 9, 9, 1) final patch

    return tf.keras.Model(inputs = [inp, tar], outputs = last)#puts the whole process into a callable model

discriminator = Discriminator()

LAMBDA = 100#lambda value for L1 loss

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discriminator_loss(disc_real_output, disc_generated_output):#calculates our loss function for our discriminator (evaluates how good our image is)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)#calculates loss for real image
    #sigmoid cross entropy between the real image and an array of ones of the same shape
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)#calculates loss for generated image

    total_disc_loss = real_loss + generated_loss#finds total loss

    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):#calculates our loss for our generator
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)#calculates loss for our generated image
    #sigmoid cross entropy between our generated image and an array of ones
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))#calculates our L1 loss of mean absolute error

    mse = tf.losses.mean_squared_error(target, gen_output)
    #helps our image become structurally similar to our target image
    total_gen_loss = gan_loss + l1_loss + mse#calculates our total loss

    return total_gen_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-3, beta_1 = 0.5, beta_2 = 0.999)#optimizer for generator
discriminator_optimizer = tf.keras.optimizers.Adam(7e-4, beta_1 = 0.5, beta_2 = 0.999)#optimizer for discriminator

checkpoint_dir = './training_checkpoints'#establishes a checkpoint directory
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")#establishes a prefix for our checkpoints
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer, discriminator_optimizer= discriminator_optimizer, generator = generator, discriminator = discriminator)
#tells us what we want saved in our checkpoints
EPOCHS = 250# number of epochs

def generate_images(model1, model2, test_input, tar, filename):#will generate our images from our test set
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model1(test_input, training=False)#makes our prediction from our test_input
    prediction = tf.where(prediction<=0.95,0.0,prediction)#cleans images and puts pixels at only values of 0 and 1
    prediction = tf.where(prediction>0.95,1.0,prediction)
    discriminator = model2([tar, prediction], training=False)#runs our prediction through the discriminator with the test_input
    #disc_loss = discriminator_loss(tar, prediction).numpy()#calculates loss value from discriminator
    mse = np.sum(tf.losses.mean_squared_error(tar, prediction).numpy())/(112**2)#calculates the mean squared error on our image
    plt.figure(figsize=(15,5))#figure size
    display_list = [test_input[0], tar[0], prediction[0]]#the images to be displayed
    title = ['Input Image', 'Ground Truth', 'Predicted Image, MSE:' +  str(mse), 'Discriminator Image']#title per subplot

    for i in range(3):#creates our subplot
        plt.subplot(1, 4, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.subplot(1,4,4)#plots discriminator image on 4th subplot
    plt.title(title[3])
    plt.imshow(discriminator[0,...,-1], cmap = 'RdBu_r')
    plt.colorbar()
    #plt.subplot(1,5,5)
    #plt.title(title[4])
    #plt.imshow(tf.losses.mean_squared_error(prediction, tar).numpy()[0])
    plt.savefig(filename + '.png')#saves figure
    return mse

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training = True)#generates image from input image and trains the generator
        #gen_output1 = tf.where(gen_output<=0.5, 0.0, gen_output)#cleans images
        #gen_output1 = tf.where(gen_output>0.5, 1.0, gen_output)
        mse = tf.losses.mean_squared_error(target, gen_output)#calculates the mean squared error between the real and generated images
        disc_real_output = discriminator([input_image, target], training = True)#runs the discriminator on the real output
        disc_generated_output = discriminator([input_image, gen_output], training = True)#runs the discriminator on the generated output
        gen_loss = generator_loss(disc_generated_output, gen_output, target)#calculates loss on the generator from the generated discriminator output, the output of the generator, and the target image
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)#calculates the discriminator loss using the real image and generated image
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)#calculates gradients using the generator loss
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)#calculates gradients using the loss from the discriminator

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))#applies gradients to the optimizer and changes variables accordingly
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))#applies gradients to the optimizer and changes variables accordingly
    return mse , gen_loss, disc_loss

def test(dataset, model1, model2):
    mse_list = []
    gen_loss_list = []
    disc_loss_list = []
    for image, target in dataset:
        gen_output = model1(image, training = True)
        disc_gen_output = model2([target, gen_output], training = True)
        disc_real_output = model2([target, image], training = True)
        mse = tf.losses.mean_squared_error(target,gen_output).numpy()
        disc_loss = discriminator_loss(disc_real_output, disc_gen_output)
        gen_loss = generator_loss(disc_gen_output, gen_output, target).numpy()
        mse_list.append(mse)
        gen_loss_list.append(gen_loss)
        disc_loss_list.append(disc_loss)
    mse_avg = np.average(np.concatenate(mse_list))
    gen_loss_avg = np.average(np.concatenate(gen_loss_list))
    disc_loss_avg = np.average(disc_loss_list)
    return mse_avg, gen_loss_avg, disc_loss_avg

def train(train_dataset, val_dataset, epochs):#trains on the training dataset for a set number of epochs
    mse_avg = []#contains average mse values from each epoch
    gen_loss_avg = []#contains average gen loss value from each epoch
    disc_loss_avg = []#average discriminator loss values from each epoch
    mse_val_list = []
    gen_loss_val_list = []
    disc_loss_val_list = []
    best_mse = 100
    best_epoch = 0
    limit = 20
    for epoch in range(epochs):#iterates through epochs
        start = time.time()#times each epoch
        mse_list = []#contains every mse value for every image in an epoch
        gen_loss_list = []#contains every generator loss value for every image in an epoch
        disc_loss_list = []#contains every discriminator loss value for every image in an epoch
        for input_image, target in train_dataset:#iterates through input and expected images
            mse, gen_loss, disc_loss = train_step(input_image, target)#runs the images through train_step function
            mse_list.append(mse.numpy())#appends mse value
            gen_loss_list.append(gen_loss.numpy())#appends gen loss value
            disc_loss_list.append(disc_loss.numpy())#appends disc loss value
        clear_output(wait = True)#clears outputs
        mse_avg.append(np.average(np.concatenate(mse_list)))#averages mse values per epoch
        gen_loss_avg.append(np.average(np.concatenate(gen_loss_list)))#averages gen loss values per epoch
        disc_loss_avg.append(np.average(disc_loss_list))#averages disc loss values per epoch
        mse_val, gen_loss_val, disc_loss_val = test(val_dataset,generator,discriminator)
        mse_val_list.append(mse_val)
        gen_loss_val_list.append(gen_loss_val)
        disc_loss_val_list.append(disc_loss_val)
        if (epoch + 1) % 5 == 0:#saves checkpoints every ten epochs
            #checkpoint.save(file_prefix = checkpoint_prefix)
            for inp, tar in test_dataset.take(1):
                generate_images(generator, discriminator, inp, tar, 'training_epoch' + str(epoch+1))
        if mse_val_list[epoch] < best_mse:
            best_mse = mse_val_list[epoch]
            best_epoch = epoch + 1
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n' .format(epoch + 1, time.time()-start))#prints time taken per epoch
        print('Average Generator Loss:' + str(gen_loss_avg[epoch]))
        print('Average Discriminator Loss:' + str(disc_loss_avg[epoch]))
        print('Average Validation Generator Loss:' + str(gen_loss_val_list[epoch]))
        print('Average Validation Discriminator Loss:' + str(disc_loss_val_list[epoch]))
        print('Best epoch:' + str(best_epoch))
        print(mse_val_list[epoch])
        if (epoch+1)-best_epoch >= limit:
            print("There was no improvement since epoch " + str(best_epoch) + ". Training has been stopped.")
            break
    return mse_avg, gen_loss_avg, disc_loss_avg, mse_val_list, gen_loss_val_list, disc_loss_val_list

def plot():
    mse_avg, gen_loss_avg, disc_loss_avg, mse_val_list, gen_loss_val_list, disc_loss_val_list = train(train_dataset, val_dataset, EPOCHS)#takes mse, and loss metrics from the train function
    plt.figure(figsize = (15,15))#sets figure size
    plt.subplot(3,1,1)#first subplot
    plt.plot(gen_loss_avg, label = 'Avg Generator Training Loss')#plots average generator loss per epoch
    plt.plot(gen_loss_val_list, label = 'Avg Generator Validation Loss')
    plt.legend(loc = 'lower right')
    plt.ylabel('Generator Cross entropy')
    plt.xlabel('Epoch')
    plt.title('Average Generator Loss Per Epoch')

    plt.subplot(3,1,2)
    plt.plot(disc_loss_avg, label = 'Avg Discriminator Training Loss')#plots average discriminator loss per epoch
    plt.plot(disc_loss_val_list, label = 'Avg Discriminator Validation Loss')
    plt.legend(loc = 'upper right')
    plt.ylabel('Discriminator Cross Entropy')
    plt.xlabel('Epoch')
    plt.title('Average Discriminator Loss Per Epoch')

    plt.subplot(3,1,3)#second subplot
    plt.plot(mse_avg, label = "Avg Training Mean Squared Error Per Pixel")#plots the average mse per pixel per epoch
    plt.plot(mse_val_list, label = 'Avg Validation Mean Squared Error')
    plt.legend(loc = 'upper right')
    plt.ylabel('Mean Squared Error per Pixel')
    plt.xlabel('Epoch')
    plt.title('Average Mean Squared Error per Pixel per Epoch')

    plt.savefig("Pix2Pix_metrics")#saves figure

plot()
#train(train_dataset, EPOCHS)#trains Pix2Pix
#
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#checkpoint.restore('./training_checkpoints/ckpt-20')

#for i in range(20):
 #   inp = tf.expand_dims(event_test[i],0)
  #  tar = tf.expand_dims(track_test[i],0)
   # generate_images(generator, discriminator, inp, tar, i)

#print(test(val_dataset, generator, discriminator))
