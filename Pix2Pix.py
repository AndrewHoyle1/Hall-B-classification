import h5py
import glob
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, utils
import os
import time
from IPython.display import clear_output

Event = []#empty list where negative data will go
Track = []#empty list where positive data will go
for filename in sorted(glob.iglob("/home/hoyle/Hall_b/Event/*")):#iterates through negative files
    img = Image.open(filename)
    border = (0,38)
    new_img = ImageOps.expand(img,border = border)
    data = np.asarray(new_img)
    Event.append(data)#converts pixel data to numpy arrays and stores them into the list
for filename in sorted(glob.iglob("/home/hoyle/Hall_b/Track/*")):#iterates through positive files
    img = Image.open(filename)
    border = (0,38)
    new_img = ImageOps.expand(img,border = border)
    data = np.asarray(new_img)
    Track.append(data)#converts pixel data to numpy arrays and stores them into the list

BUFFER_SIZE = 400
BATCH_SIZE = 1

event, track = utils.shuffle(Event, Track, random_state = 0)
event = np.float32(event)
track = np.float32(track)
event_train, event_test, track_train, track_test = model_selection.train_test_split(event, track, test_size = 0.25, shuffle = False)

train_dataset = tf.data.Dataset.from_tensor_slices((event_train,track_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((event_test, track_test)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm = True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides = 2, padding = 'same', kernel_initializer= initializer, use_bias = False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

down_model = downsample(3,4)

def upsample(filters, size, apply_dropout = False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides = 2, padding = 'same', kernel_initializer = initializer, use_bias = False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

up_model = upsample(3,4)

def Generator():
    down_stack = [
        downsample(64,4,apply_batchnorm=False),# 56,56,64
        downsample(128,4),# 28,28,128
        downsample(256,4),# 14, 14, 256
        downsample(512,4),# 7,7, 512
    ]

    up_stack = [
        upsample(256, 4, apply_dropout = True),
        upsample(128,4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides = 2, padding = 'same', kernel_initializer = initializer, activation = 'tanh')

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape = [None, None, 3])
    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x,skip])

    x = last(x)

    return tf.keras.Model(inputs = inputs, outputs = x)

generator = Generator()

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape = [None, None, 3], name = 'input_image')
    tar  = tf. keras.layers.Input(shape = [None, None, 3], name = 'target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64,4,False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
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
        plt.subplot(1,3, i+1)
        plt.title(title[i])

        plt.imshow(display_list[i] * 0.5 + 0.5)
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

#for inp, tar in test_dataset.take(5):
    #generate_images(generator, inp, tar)
