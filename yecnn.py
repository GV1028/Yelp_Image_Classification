import csv
import nltk
import numpy as np
import itertools
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from PIL import Image
from scipy import misc
from skimage import color
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Convolution2D(32,5,5, border_mode='valid',input_shape=(3, 100, 100))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 5, 5, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64*5*5, init='normal'))
model.add(Activation('relu'))

model.add(Dense(100, init='normal'))
model.add(Activation('relu'))

model.add(Dense(9, init='normal'))
model.add(Activation('softmax'))
model.summary()
sgd = SGD(l2=0.001,lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

def rimage(file):
	path1 = "/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/train_photos/"
	im = misc.imread(path1 + file + ".jpg")
	#print im.shape
	im = misc.imresize(im,(100,100,3))
	im = np.transpose(im, (2, 0, 1))
	return im

def returnimage(file,no,batch):
	x = np.zeros((1000,3,100,100))
	j = 0
	for i in range(no*batch,no*(batch+1)) :
			x[j,:,:,:] = rimage(file[i])
			print ("Reading Image %d" %i)
			j =j+1
	return x

def returnimagelast(file,no,batch):
	x = np.zeros((842,3,100,100))
	j = 0
	for i in range(234000+batch,no+batch+234000-1) :
			x[j,:,:,:] = rimage(file[i])
			print ("Reading Image %d" %i)
			j =j+1
	return x
def returnylast():
	with open('/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/trainfinal.csv','rb') as f:
		reader = csv.reader(f, skipinitialspace=True)
		reader.next()	
		y = [x[2:11] for x in reader]
	y_train = y[234000:234843]
	y = np.asarray(y_train)
	return y_train

def getdata(no,batch):

	with open('/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/trainfinal.csv','rb') as f:
		reader = csv.reader(f, skipinitialspace=True)
		reader.next()	
		y = [x[2:11] for x in reader]
	with open('/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/trainfinal.csv','rb') as f:
		reader = csv.reader(f, skipinitialspace=True)
		reader.next()			
		id = [x[0] for x in reader]
	
	x_train = returnimage(id,no,batch)
	y_train = y[no*batch:no*(batch+1)]
	y_train = np.asarray(y_train)
	return x_train,y_train

datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse')
for e in range(234):
	x,y = getdata(1000,e)
	x = x.astype('float32')
	x = x/255
	model.fit(x,y,nb_epoch = 30)

with open('/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/trainfinal.csv','rb') as f:
		reader = csv.reader(f, skipinitialspace=True)
		reader.next()	
		y = [x[2:11] for x in reader]
	
with open('/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/trainfinal.csv','rb') as f:
		reader = csv.reader(f, skipinitialspace=True)
		reader.next()			
		id = [x[0] for x in reader]

x = returnimagelast(id,843,0)
y = returnylast()	
x = x.astype('float32')
x = x/255
model.fit(x,y,nb_epoch = 30)
model.save_weights("/home/vignesh/Desktop/Projects/yelpweightssoftmax.h5",overwrite=False)
