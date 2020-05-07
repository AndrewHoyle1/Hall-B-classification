import h5py
import tensorflow as tf
import numpy as np
from sklearn import model_selection, utils

print("Imported for data management")


def Get_data():
    f = h5py.File('Pix2Pix_oneTrack', 'r')
    Event = f.get('all')
    Event = np.array(Event)
    Track = f.get('truth')
    Track = np.array(Track)

    BUFFER_SIZE = 1000
    BATCH_SIZE = 256

    event, track = utils.shuffle(Event, Track, random_state = 0)#shuffles event and track data in unison
    event = np.float32(event)/255#converts event data to float32
    track = np.float32(track)/255#converts track data to float32
    event_train, event_test, track_train, track_test = model_selection.train_test_split(event, track, test_size = 0.25, shuffle = False)#splits into training and testing sets
    event_train, event_val, track_train, track_val = model_selection.train_test_split(event_train, track_train, test_size = 0.25, shuffle = False)

    train_dataset = tf.data.Dataset.from_tensor_slices((event_train,track_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)#creates tensorflow training dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((event_test, track_test)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)#creates tensorflow testing dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((event_val, track_val)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)

    return event_train, event_test, track_test, train_dataset, test_dataset, val_dataset
