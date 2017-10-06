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

from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D

import keras as k
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

from keras.models import Model
from keras.layers import Input
from keras.layers import merge
from keras.models import load_model
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D

def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')
def find_f2score_threshold(e, p_valid, y_valid, try_all=False, verbose=False):
    best = 0
    best_score = -1
    totry = np.arange(0,1,0.005) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = f2_score(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print(e, ' Best score: ', round(best_score, 5), ' @ threshold =', best)
    return best_score


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
nsamples = len(df_train)
split = int(.9*nsamples)

img_input = Input(shape=(128,128,3))
# Block 1
b1s1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
b1s2 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(b1s1)
b1s3 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(b1s2)

b1o = GlobalAveragePooling2D()(b1s3)

# Block 2
b2s1 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(b1s3)
b2s2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(b2s1)
b2s3 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(b2s2)

b2o = GlobalAveragePooling2D()(b2s3)

# Block 3
b3s1 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(b2s3)
b3s2 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(b3s1)
b3s3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(b3s2)
b3s4 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(b3s3)

b3o = GlobalAveragePooling2D()(b3s4)

# Block 4
b4s1 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(b3s4)
b4s2 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(b4s1)
b4s3 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(b4s2)
b4s4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(b4s3)

b4o = GlobalAveragePooling2D()(b4s4)

# Block 5
b5s1 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(b4s4)
b5s2 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(b5s1)
b5s3 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(b5s2)
b5s4 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(b5s3)

b5o = GlobalAveragePooling2D()(b5s4)

# Block 6
b6s1 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(b5s4)
b6s2 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(b6s1)
b6s3 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(b6s2)
b6s4 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(b6s3)

b6o = GlobalAveragePooling2D()(b6s4)

# Block 7
b7s1 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(b6s4)
b7s2 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(b7s1)
b7s3 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(b7s2)
b7s4 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(b7s3)

b7o = GlobalAveragePooling2D()(b7s4)

#keras.layers.merge.Concatenate(axis=-1)
con = merge.Concatenate([b1o,b2o,b3o,b4o,b5o,b6o,b7o], axis=1)
dense1 = Dense(1024, activation='relu')(con)
predictions = Dense(13, activation='sigmoid')(dense1)
model = Model(img_input, predictions, name='g4')
model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])

h5f = h5py.File('./dataset/valdata','r')
valfeatures = h5f['features'][:]
vallabels = h5f['labels'][:]

best = 0
batch_size = 50
val_size = nsamples - split
trainfile = './dataset/traindata'
train_generator = generator(trainfile, split, batch_size)
for e in range(1):
  model.fit_generator(train_generator, samples_per_epoch=batch_size, nb_epoch=1)#, validation_data=(valfeatures, vallabels), nb_val_samples=val_size)
  if e%5 == 0:
    p_valid = model.predict(valfeatures, batch_size=128)
    temp = find_f2score_threshold(e, p_valid, vallabels, verbose=True) 
  	if temp > best:
  		model.save('g4model_best.h5') 
		best = temp
  	else:
		print "Not the best