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

sgd = SGD(l2=0.001,lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.load_weights("/home/vignesh/Desktop/Projects/yelpweightssoftmax.h5")

print "Model Loaded"
def rimage(file):
	path1 = "/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/Yelp_Test/test_photos/"
	im = misc.imread(path1 + file + ".jpg")
	#print im.shape
	
	im = misc.imresize(im,(100,100,3))
	im = np.transpose(im, (2, 0, 1))
	return im

def returnimage(file):
	x = np.zeros((1,3,100,100))
	x = rimage(id)	
	return x

with open('/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/id.csv','rb') as f:
	reader = csv.reader(f, skipinitialspace=True)
	reader.next()			
	id = [x[0] for x in reader]
	
print len(id)

for i in range(len(id)):
	x = rimage(id[i])
	print "Testing image %d" %i
	x = x.astype('float32')
	x = x/255	
	x = np.expand_dims(x, axis=0)
	preds = model.predict(x)
	preds = preds[0]
	with open('/media/vignesh/Windows/Users/VIGNESH/Videos/DATA_SCIENCE/Kaggle/Yelp/cnnrelu.csv','ab') as file:
		fieldnames = ['C0', 'C1','C2','C3','C4','C5','C6','C7','C8']
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writerow({'C0': preds[0],'C1': preds[1],'C2': preds[2],'C3': preds[3],'C4': preds[4],'C5': preds[5],'C6': preds[6],'C7': preds[7],'C8': preds[8]})
