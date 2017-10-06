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
from keras.applications.resnet50 import ResNet50
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
i=0
nsamples = len(df_train)
split = int(.9*nsamples)

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

def generator(filename, batch_size):
  batch_features = np.zeros((batch_size, 256,256,3), np.float16)
  batch_labels = np.zeros((batch_size,17))
  h5f = h5py.File(filename,'r')
  size = h5f['features'].size/(256*256*3)
  while True:
    for i in range(batch_size):
       # choose random index in features
       index= np.random.randint(1, size)
       img = h5f['features'][index]
       targets = h5f['labels'][index]
       batch_features[i] = img
       batch_features[i] = batch_features[i]/255
       batch_labels[i] = targets
    yield ({'input':batch_features}, {'output_1':batch_labels[:14],'output_2':batch_lables[14:]})

def valresults(filename, model):
  h5f = h5py.File(filename,'r')
  size = h5f['features'].size/(256*256*3)
  results = np.zeros((size,17))
  batch_features = np.zeros((500, 256,256,3), np.float16)
  #batch_labels = h5f['labels'][:]
  for index in range(0, size, 500):
     # choose random index in features
     batch_features = h5f['features'][index:index+500]
     results[index:index+500] = model.predict(batch_features, batch_size=100)
  return results

base_model = ResNet50(weights='imagenet', include_top=False)
#base_model = VGG16(weights='imagenet', include_top=False)
b = base_model.output
b = GlobalAveragePooling2D()(b)
b = Dense(1024, activation='relu')(b)
predictions1 = Dense(13, activation='sigmoid')(b)
predictions2=Dense(4,activation='softmax')(b)
model = Model(input=base_model.input, output=[predictions1,predictions2])
model.compile(loss=['binary_crossentropy','categorical_crossentropy'], # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
				metrics=['accuracy'])

#model = load_model('model256_epoch_50.h5')
callbacks = [EarlyStopping(monitor='val_acc',patience=20,verbose=0)]
batch_size = 40
trainfile = 'traindata256'
valfile = 'valdata256'
h5f = h5py.File(valfile,'r')
vallabels = h5f['labels'][:]
h5f.close()
best = 0
train_generator = generator(trainfile, batch_size)
for e in range(50, 500):

  model.fit_generator(train_generator, samples_per_epoch=batch_size, nb_epoch=5)
  p_valid = valresults(valfile, model)
  temp = find_f2score_threshold(e, p_valid, vallabels, verbose=True) 
  if temp > best:
  	model.save('ResnetBest256.h5') 
	best = temp
  else:
	print "Not the best"

model.save('Resnetmodel256_epoch_final' + str(e) +'.h5')  
print e, (fbeta_score(vallabels, np.array(p_valid) > 0.2, beta=2, average='samples'))


'''
for f, tags in tqdm(df_train.values, miniters=1000):
    i+=1
    if (i>nsamples):
        break
    img = cv2.imread('./train-jpg/{}.jpg'.format(f),cv2.COLOR_BGR2RGB)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (256, 256)))
    y_train.append(targets)

x_train, y_train=shuffle(x_train, y_train)
#create dataset
h5f = h5py.File('traindata', 'w')
h5f.create_dataset('features', data=x_train[:split], chunks = True)
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


labels_df = pd.read_csv('./input/train_v2.csv')

label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

# Add onehot features for every label
for label in label_list:
    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# Display head
labels_df.head()

df = labels_df[(labels_df.blow_down == 1) | (labels_df.conventional_mine == 1) | 
(labels_df.slash_burn ==1) | (labels_df.blooming==1) | (labels_df.artisinal_mine == 1) | 
(labels_df.selective_logging == 1) | (labels_df.bare_ground == 1) | (labels_df.cloudy == 1) | 
(labels_df.haze == 1) | (labels_df.habitation == 1) | (labels_df.cultivation == 1) ]

import matplotlib.pyplot as plt
df[label_list].sum().sort_values().plot.bar()
plt.show()

for x in df.itertuples():
  cmd = 'cp input/train-jpg/' + x.image_name +'.jpg'+ ' image_augmentor/image/.'
  os.system(cmd)

import glob
imgs = glob.glob("image_augmentor/image/*") 

for imgname in imgs:
  img = cv2.imread(imgname,cv2.COLOR_BGR2RGB)
  cv2.imwrite(imgname, img)

newfile = open('subset.txt','w')
newfile.write('image_name,tags\n')
for x in df.itertuples():
   string = x.image_name+','+x.tags+'\n'
   newfile.write(string)
newfile.close()

python2.7 main.py image/ fliph flipv rot_30 rot_60 trans_20_10

newfile = open('subset.txt','w')
newfile.write('image_name,tags\n')
for x in df.itertuples():
    for a in ['', '__fliph','__flipv', '__rot30', '__rot60', '__trans20_10']: 
        string = x.image_name+a+','+x.tags+'\n'
        newfile.write(string)
newfile.close()

'''
