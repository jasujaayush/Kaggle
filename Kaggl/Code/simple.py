import os
import gc
import cv2
import h5py
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from random import randint
from sklearn.utils import shuffle
from sklearn.metrics import fbeta_score

import keras as k
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D

def generator(filename, split, batch_size):
  batch_features = np.zeros((batch_size, 128,128,3), np.float32)
  batch_labels = np.zeros((batch_size,17))
  h5f = h5py.File(filename,'r')
  while True:
    for i in range(batch_size):
       # choose random index in features
       index= np.random.randint(1, split - 1)
       batch_features[i] = h5f['features'][index]
       batch_features[i] = batch_features[i]/255
       batch_labels[i] = h5f['labels'][index] 
    yield batch_features, batch_labels


x_train = []
y_train = []

df_train = pd.read_csv('./train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
i=0
nsamples = 0.1*len(df_train)
split = int(.9*nsamples)

for f, tags in tqdm(df_train.values, miniters=1000):
    i+=1
    if (i>nsamples):
        break
    img = cv2.imread('./train-jpg/{}.jpg'.format(f),cv2.COLOR_BGR2RGB)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 

    x_train.append(cv2.resize(img, (128, 128)))
    y_train.append(targets)

x_train, y_train=shuffle(x_train, y_train)

#create dataset
h5f = h5py.File('traindata', 'w')
h5f.create_dataset('features', data=x_train[:split], chunks=True)
h5f.create_dataset('labels', data=y_train[:split], chunks = True)
h5f.close()

h5f = h5py.File('valdata', 'w')
h5f.create_dataset('features', data=x_train[split:])
h5f.create_dataset('labels', data=y_train[split:])
h5f.close()

del x_train
del y_train

#load dataset
h5f = h5py.File('valdata','r')
valfeatures = h5f['features'][:] 
vallabels = h5f['labels'][:] 

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(128, 128, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))
model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_acc',patience=20,verbose=0)]

batch_size = 50
val_size = nsamples - split
trainfile = 'traindata'
train_generator = generator(trainfile, split, batch_size)
for e in range(500):
  model.fit_generator(train_generator, samples_per_epoch=batch_size, nb_epoch=1)#, validation_data=(valfeatures, vallabels), nb_val_samples=val_size)
  if e%5 == 0:
	p_valid = model.predict(valfeatures, batch_size=128)
    	print e+1, (fbeta_score(vallabels, np.array(p_valid) > 0.2, beta=2, average='samples'))
    #model.save('model epoch ' + str(e+1) +'.h5')  

#model.save('model epoch ' + str(e+1) +'.h5')  
#p_valid = model.predict(valfeatures, batch_size=128)
#print(fbeta_score(vallabels, np.array(p_valid) > 0.2, beta=2, average='samples'))
#print(y_valid)
#print(p_valid)
