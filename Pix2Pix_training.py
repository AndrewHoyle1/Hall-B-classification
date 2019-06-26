import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, utils
import os
import time
from IPython.display import clear_output

f = h5py.File('Pix2Pix_data', 'r')
Event = f.get('event')
Track = f.get('track')

BUFFER_SIZE = 400
BATCH_SIZE = 1

event, track = utils.shuffle(Event, Track, random_state = 0)#shuffles event and track data in unison
event = np.float32(event)#converts event data to float32
track = np.float32(track)#converts track data to float32
event_train, event_test, track_train, track_test = model_selection.train_test_split(event, track, test_size = 0.25, shuffle = False)#splits into training and testing sets

train_dataset = tf.data.Dataset.from_tensor_slices((event_train,track_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)#creates tensorflow training dataset
test_dataset = tf.data.Dataset.from_tensor_slices((event_test, track_test)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)#creates tensorflow testing dataset

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

down_model = downsample(3,4)

def upsample(filters, size, apply_dropout = False):#upsamples images by a factor of 2
    initializer = tf.random_normal_initializer(0., 0.02)#initializer

    result = tf.keras.Sequential()#sequential model for new layers
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides = 2, padding = 'same', kernel_initializer = initializer, use_bias = False))#convolutional transpose layer where we upsize

    result.add(tf.keras.layers.BatchNormalization())#batch normalization

    if apply_dropout:#applies dropout when instructed (dropout essentially selects random neurons to ignore when training)
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())#applies ReLU

    return result

up_model = upsample(3,4)

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

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape = [None, None, 3], name = 'input_image')#where we will assign our input image
    tar  = tf. keras.layers.Input(shape = [None, None, 3], name = 'target_image')#where we will assign our target image

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64,4,False)(x)#downsamples once for patchGAN
    #down2 = downsample(128, 4)(down1)
    #down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down1)#zero pads for convolutional layer
    conv = tf.keras.layers.Conv2D(512,4,strides = 1, kernel_initializer = initializer, use_bias = False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1,4, strides = 1, kernel_initializer= initializer)(zero_pad2)

    return tf.keras.Model(inputs = [inp, tar], outputs = last)

discriminator = Discriminator()

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss+generated_loss

    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer, discriminator_optimizer= discriminator_optimizer, generator = generator, discriminator = discriminator)

EPOCHS = 150

def generate_images(model, test_input, tar):
    prediction = model(test_input, training = True)
    plt.figure(figsize = (15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(j+1,3,i+1)
        plt.title(title[i])

        plt.imshow(display_list[i])
        plt.axis('off')
    plt.savefig('testing_attempt.png')

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training = True)

        disc_real_output = discriminator([input_image, target], training = True)
        disc_generated_output = discriminator([input_image, gen_output], training = True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for input_image, target in dataset:
            train_step(input_image, target)

        clear_output(wait = True)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n' .format(epoch + 1, time.time()-start))

train(train_dataset, EPOCHS)

#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
