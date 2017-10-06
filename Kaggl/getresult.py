import os
import gc
import cv2
import h5py
import os.path
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from random import randint
from sklearn.utils import shuffle
from sklearn.metrics import fbeta_score

import keras as k
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D

  
x_train = []
y_train = []

df_train = pd.read_csv('./train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

model = load_model('model epoch 61.h5')
result = 'image_name,tags\n'
batch = 500
imsize = 128

path = './test-jpg/'
n = len(os.listdir(path))
x_test = np.zeros((batch, imsize,imsize,3), np.float16)
f = 0
tf = 0
while True:
	if (f>0 and f%batch == 0) or (f == n):
		print f
		p_valid = model.predict(x_test, batch_size=100)
		x_test = np.zeros((batch, imsize,imsize,3), np.float16)
		for t in range(0, batch):
			tags = ''
			tarray = np.where(p_valid[t]>0.2)[0]
			for p in tarray:
				tags = tags + inv_label_map[p] + ' '
			result = result + 'test_{},'.format(tf)+tags + '\n'
			tf += 1
	if f==n:
		break	
	img = cv2.imread('./test-jpg/test_{}.jpg'.format(f),cv2.COLOR_BGR2RGB)
	x_test[f%batch] = cv2.resize(img, (128, 128))
	f = f + 1

path = './test-jpg-additional/'
n = len(os.listdir(path))
x_test = np.zeros((batch, imsize,imsize,3), np.float16)
f = 0
tf = 0
while True:
	if (f>0 and f%batch == 0) or (f == n):
		p_valid = model.predict(x_test, batch_size=100)
		x_test = np.zeros((batch, imsize,imsize,3), np.float16)
		for t in range(0, batch):
			tags = ''
			tarray = np.where(p_valid[t]>0.2)[0]
			for p in tarray:
				tags = tags + inv_label_map[p] + ' '
			result = result + 'file_{},'.format(tf)+tags + '\n'
			tf += 1
	if f==n:
		break	
	img = cv2.imread('./test-jpg-additional/file_{}.jpg'.format(f),cv2.COLOR_BGR2RGB)
	x_test[f%batch] = cv2.resize(img, (128, 128))
	f = f + 1

fi = open('results.txt','w')    
fi.write(result)
fi.close()
