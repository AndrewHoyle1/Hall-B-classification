import h5py
import glob
import numpy as np
from PIL import Image, ImageOps


f = h5py.File('Pix2Pix_data')

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

d_event = f.create_dataset('event', data = Event)
d_track = f.create_dataset('track', data = Track)
