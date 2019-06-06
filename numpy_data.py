import h5py
from matplotlib import image
import glob
import  numpy as np
from PIL import Image, ImageOps


f = h5py.File('Num_data')#calls file where we will store the numpy versions of our image data

data_neg = []#empty list where negative data will go
data_pos = []#empty list where positive data will go
for filename in glob.iglob("/home/anhoyle/Hall_b/dctrack_neg/*"):#iterates through negative files
    img = Image.open(filename)
    border = (0,38)
    new_img = ImageOps.expand(img,border = border)
    data = np.asarray(new_img)
    data_neg.append(data)#converts pixel data to numpy arrays and stores them into the data_neg list
for filename in glob.iglob("/home/anhoyle/Hall_b/dctrack_pos/*"):#iterates through positive files
    img = Image.open(filename)
    border = (0,38)
    new_img = ImageOps.expand(img,border = border)
    data = np.asarray(filename)
    data_pos.append(data)#converts pixel data to numpy arrays and stores them into the data_pos list
dset1 = f.create_dataset('neg', data = data_neg)#stores numpy data into a dataset
dset2 = f.create_dataset('pos', data = data_neg)#stores numpy data into a dataset
