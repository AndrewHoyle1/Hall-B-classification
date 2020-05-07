import h5py
import glob
import numpy as np
from PIL import Image, ImageOps
#from matplotlib import pyplot as plt

f = h5py.File('Pix2Pix_oneTrack')

Event = []#empty list where negative data will go
Track = []#empty list where positive data will go
for filename in sorted(glob.iglob("/home/hoyle/Hall_b/oneTrack/*_all_*")):#iterates through negative files
    img = Image.open(filename)
    section_border = np.zeros((1,128,3))
    border = (8,45)
    img = ImageOps.expand(img, border)
    data = np.asarray(img)
    section1 = data[0:56,:,:]
    section1 = np.concatenate((section1,section_border))
    section2 = data[57:69,:,:]
    section2 = np.concatenate((section_border,section2,section_border))
    section3 = data[70:126,:,:]
    section3 = np.concatenate((section_border,section3))
    new_img = np.concatenate((section1,section2,section3))
    
    Event.append(new_img)
    

for filename in sorted(glob.iglob("/home/hoyle/Hall_b/oneTrack/*_truth_*")):#iterates through positive files
    img = Image.open(filename)
    section_border = np.zeros((1,128,3))
    border = (8,45)
    img = ImageOps.expand(img, border)
    data = np.asarray(img)
    section1 = data[0:56,:,:]
    section1 = np.concatenate((section1,section_border))
    section2 = data[57:69,:,:]
    section2 = np.concatenate((section_border,section2,section_border))
    section3 = data[70:126,:,:]
    section3 = np.concatenate((section_border,section3))
    new_img = np.concatenate((section1,section2,section3))
    
    Track.append(new_img)
    

print(len(Event))
print(len(Track))
d_event = f.create_dataset('all', data = Event)
d_track = f.create_dataset('truth', data = Track)
