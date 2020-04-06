import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from IPython import display
from data_management import Get_data
import datetime

event_train, event_test, track_test, train_dataset, test_dataset, val_dataset = Get_data()

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
        downsample(64,4,apply_batchnorm=False),# layers of size (bs, 64,64,64)
        downsample(128,4),# (bs, 32,32, 128)
        downsample(256,4),# (bs, 16, 16, 256)
        downsample(512,4),# (bs, 8, 8, 512)
        downsample(512,4),# (bs, 4,4,512)
        downsample(512,4),# (bs, 2, 2, 512)
        downsample(512,4) # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512,4, apply_dropout = True),# (bs, 2, 2, 1024)
        upsample(512,4, apply_dropout = True), #(bs, 4, 4, 1024)
        upsample(512,4, apply_dropout = True), #(bs, 8, 8, 1024)
        upsample(256,4), #(bs, 16, 16, 512)
        upsample(128,4), #(bs, 32, 32, 256)
        upsample(64, 4), #(bs, 64, 64, 128)

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

def Discriminator():#analyzes our images and evaluates how good they are
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape = [None, None, 3], name = 'input_image')#where we will assign our input image
    tar  = tf. keras.layers.Input(shape = [None, None, 3], name = 'target_image')#where we will assign our target image

    x = tf.keras.layers.concatenate([inp, tar])#concatenates our images (bs, 112,112,channels*2)

    down1 = downsample(64,4,False)(x)#downsamples (bs, 56, 56, 64)
    down2 = downsample(128, 4)(down1)#downsamples (bs, 28, 28, 128)
    down3 = downsample(256, 4)(down2)#downsamples (bs, 14, 14, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)#zero pads our most recent layer (bs, 16, 16, 256)
    conv = tf.keras.layers.Conv2D(512,4,strides = 1, kernel_initializer = initializer, use_bias = False)(zero_pad1)#(bs, 12, 12, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)# does batchnormalization on our most recent layer

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)#applies a leaky ReLU function to our batch normalized layer

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)#zero pads our last layer (bs, 13, 13, 512)

    last = tf.keras.layers.Conv2D(1,4, strides = 1, kernel_initializer= initializer)(zero_pad2)#(bs, 9, 9, 1) final patch

    return tf.keras.Model(inputs = [inp, tar], outputs = last)#puts the whole process into a callable model

generator = Generator()
discriminator = Discriminator()

LAMBDA = 200#lambda value for L1 loss

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)
mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()

def discriminator_loss(disc_real_output, disc_generated_output):#calculates our loss function for our discriminator (evaluates how good our image is)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)#calculates loss for real image
    #sigmoid cross entropy between the real image and an array of ones of the same shape
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)#calculates loss for generated image

    total_disc_loss = real_loss + generated_loss#finds total loss

    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):#calculates our loss for our generator
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)#calculates loss for our generated image
    #sigmoid cross entropy between our generated image and an array of ones
    #l1_loss = tf.reduce_mean(tf.abs(target - gen_output))#calculates our L1 loss of mean absolute error
    mae = mae_loss(target, gen_output)
    #helps our image become structurally similar to our target image
    total_gen_loss = gan_loss  + LAMBDA*mae#calculates our total loss

    return total_gen_loss, gan_loss, mae

def generate_images(model1, model2, test_input, tar, filename):#will generate our images from our test set
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model1(test_input, training=False)#makes our prediction from our test_input
    #prediction = tf.where(prediction<=0.95,0.0,prediction)#cleans images and puts pixels at only values of 0 and 1
    #prediction = tf.where(prediction>0.95,1.0,prediction)
    discriminator = model2([tar, prediction], training=False)#runs our prediction through the discriminator with the test_input
    #disc_loss = discriminator_loss(tar, prediction).numpy()#calculates loss value from discriminator
    mse = mse_loss(tar, prediction).numpy()#calculates the number of incorrect pixels in our image
    mse *= 128**2
    #mse = np.sqrt(mse)
    plt.figure(figsize=(15,5))#figure size
    display_list = [test_input[0], tar[0], prediction[0]]#the images to be displayed
    title = ['Input Image', 'Ground Truth', 'Predicted Image, MSD:' +  str(mse), 'Discriminator Image']#title per subplot

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

generator_optimizer = tf.keras.optimizers.Adam(2e-3, beta_1 = 0.5)#optimizer for generator
discriminator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1 = 0.5)#optimizer for discriminator

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir + 'fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
checkpoint_dir = './training_checkpoints'#establishes a checkpoint directory
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")#establishes a prefix for our checkpoints
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer, discriminator_optimizer= discriminator_optimizer, generator = generator, discriminator = discriminator)
#tells us what we want saved in our checkpoints
EPOCHS = 250# number of epochs

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training = True)#generates image from input image and trains the generator
        #gen_output1 = tf.where(gen_output<=0.5, 0.0, gen_output)#cleans images
        #gen_output1 = tf.where(gen_output>0.5, 1.0, gen_output)
        #mse = mse_loss(target, gen_output)#calculates the mean squared error between the real and generated images
        disc_real_output = discriminator([input_image, target], training = True)#runs the discriminator on the real output
        disc_generated_output = discriminator([input_image, gen_output], training = True)#runs the discriminator on the generated output
        gen_total_loss, gen_gan_loss, gen_mse_loss = generator_loss(disc_generated_output, gen_output, target)#calculates loss on the generator from the generated discriminator output, the output of the generator, and the target image
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)#calculates the discriminator loss using the real image and generated image
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)#calculates gradients using the generator loss
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)#calculates gradients using the loss from the discriminator

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))#applies gradients to the optimizer and changes variables accordingly
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))#applies gradients to the optimizer and changes variables accordingly

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step = epoch)
        tf.summary.scalar('gen_mse_loss', gen_mse_loss, step = epoch)
        tf.summary.scalar('disc_loss', disc_loss, step = epoch)
    return gen_total_loss, disc_loss

def val_test(input_image, target, epoch):
    gen_output = generator(input_image, training = True)#generates image from input image and trains the generator
    #gen_output1 = tf.where(gen_output<=0.5, 0.0, gen_output)#cleans images
    #gen_output1 = tf.where(gen_output>0.5, 1.0, gen_output)
    #mse = mse_loss(target, gen_output)#calculates the mean squared error between the real and generated images
    disc_real_output = discriminator([input_image, target], training = True)#runs the discriminator on the real output
    disc_generated_output = discriminator([input_image, gen_output], training = True)#runs the discriminator on the generated output
    gen_total_loss, gen_gan_loss, gen_mse_loss = generator_loss(disc_generated_output, gen_output, target)#calculates loss on the generator from the generated discriminator output, the output of the generator, and the target image
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)#calculates the discriminator loss using the real image and generated image

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_val_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_val_loss', gen_gan_loss, step = epoch)
        tf.summary.scalar('gen_mse_val_loss', gen_mse_loss, step = epoch)
        tf.summary.scalar('disc_val_loss', disc_loss, step = epoch)
    return gen_total_loss, disc_loss

def fit(train_dataset, val_dataset, test_dataset, epochs):#trains on the training dataset for a set number of epochs
    best_epoch = 0
    best_loss = 10000
    limit = 20
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait = True)
        gen_loss_list = []
        gen_val_loss_list = []
        disc_loss_list = []
        disc_val_loss_list = []

        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, discriminator, example_input, example_target, "training_epoch" + str(epoch+1))
        print ("Epoch: ", epoch)

        for n, (input_image, target) in train_dataset.enumerate():
            print(".", end = '')
            if (n+1) % 100 == 0:
                print()
            gen_loss,disc_loss = train_step(input_image,target, epoch)
            gen_loss = gen_loss.numpy()
            disc_loss = disc_loss.numpy()
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)
        print()

        for n, (input_image, target) in val_dataset.enumerate():
            print(".", end = '')
            if (n+1) % 100 == 0:
                print()
            gen_loss,disc_loss = val_test(input_image, target, epoch)
            gen_loss = gen_loss.numpy()
            disc_loss = disc_loss.numpy()
            gen_val_loss_list.append(gen_loss)
            disc_val_loss_list.append(disc_loss)

        avg_gen_loss = np.average(gen_loss_list)
        avg_val_gen_loss = np.average(gen_val_loss_list)
        avg_disc_loss = np.average(disc_loss_list)
        avg_val_disc_loss = np.average(disc_val_loss_list)

        if (avg_gen_loss < best_loss):
            checkpoint.save(file_prefix = checkpoint_prefix + str(epoch+1))
            best_epoch = epoch + 1
            best_loss = avg_gen_loss
        elif (epoch + 1 >= best_epoch + limit):
            print("There has been no significant improvement since epoch ", best_epoch, ".")
            break

        print("Time taken for epoch {} is {} sec\n" .format(epoch+1, time.time()-start))
        print("Best epoch so far: " , best_epoch)
        print("Avg Generator Loss: ", avg_gen_loss)
        print("Avg Discriminator Loss: ", avg_disc_loss)
        print("Avg Validation Generator Loss: ", avg_val_gen_loss)
        print("Avg Validation Discriminator Loss: ", avg_val_disc_loss)

def final_test():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    good_images = 0
    mse_list = []
    for i in range(len(event_test)):
        inp = tf.expand_dims(event_test[i],0)
        tar = tf.expand_dims(track_test[i],0)
        prediction = generator(inp, training = False)
        #prediction = tf.where(prediction<=0.75, 0.0, prediction)
        #prediction = tf.where(prediction>0.75, 1.0, prediction)
        mae = mse_loss(tar, prediction).numpy()
        mae *= 128**2
        mse_list.append(mae)
        if mae <= 5:
            good_images+=1
        else:
            continue
    print("The percentage of good images is " + str(good_images/len(event_test)) + ".")
    return mse_list

def MSE_Histogram():
    mse_list = final_test()

    plt.figure(figsize = (15,15))
    plt.hist(mse_list, bins = 'auto')
    plt.ylabel('Counts')
    plt.xlabel('Pixel Difference')
    plt.savefig("MSE_Histogram")
    
def track_pixel_histogram():
    diff_list = []
    zero_array = tf.zeros((128,128,3))
    
    for i in range(len(event_train)):
        img = tf.expand_dims(event_train[i],0)
        
        mse = mse_loss(img,zero_array)
        mse*= 128**2
        diff_list.append(mse)
        
    plt.figure(figsize = (15,15))
    plt.hist(diff_list, bins = 'auto')
    plt.ylabel('Counts')
    plt.xlabel('Pixel Difference')
    plt.title('Track Pixel Distribution')
    plt.savefig('Track_pixel_histogram')
        

#fit(train_dataset, val_dataset, test_dataset, EPOCHS)

#final_test()

#MSE_Histogram()

"""def plot(train_dataset, val_dataset):
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

def final_test(model1,model2,dataset1, dataset2):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    good_images = 0
    mse_list = []
    for i in range(len(event_test)):
        inp = tf.expand_dims(event_test[i],0)
        tar = tf.expand_dims(track_test[i],0)
        mse = generate_images(generator, discriminator, inp, tar, 'test'+ str(i))
        mse_list.append(mse)
        if mse <= 4:
            good_images+=1
        else:
            continue
    plt.hist(mse_list)
    plt.savefig("MSE_Histogram")

    print("The percentage of good images is " + str(good_images/len(event_test)) + ".")

def main():
    final_test(generator, discriminator, event_test, track_test)
    #plot(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
#train(train_dataset, EPOCHS)#trains Pix2Pix

#checkpoint.restore('./training_checkpoints/ckpt-20')
#good_images = 0


#final_test(generator, discriminator, test_dataset)

#print(test(val_dataset, generator, discriminator))"""
