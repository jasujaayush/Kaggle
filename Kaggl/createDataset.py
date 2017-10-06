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
 
x_train = []
y_train = []

df_train = pd.read_csv('./train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
i=0

#create dataset
nsamples = len(df_train)
split = int(.9*nsamples)

tr = h5py.File('traindata256', 'w')
trainfeatures = tr.create_dataset("features", (split, 256, 256, 3), dtype = np.float16)
trainlabels = tr.create_dataset("labels", (split, 17))

va = h5py.File('valdata256', 'w')
valfeatures = va.create_dataset('features', (nsamples - split, 256, 256, 3), dtype = np.float16)
vallabels = va.create_dataset('labels', (nsamples - split, 17))

x_train = np.zeros((100, 256,256,3), np.float16)
y_train = np.zeros((100,17))
count = 0
vi = 0
ti = 0
for f, tags in tqdm(df_train.values, miniters=1000):
	i+=1
	if (i>nsamples):
		break
	img = cv2.imread('./train-jpg/{}.jpg'.format(f),cv2.COLOR_BGR2RGB)
	targets = np.zeros(17)
	for t in tags.split(' '):
		targets[label_map[t]] = 1 
	x_train[count] = cv2.resize(img, (256, 256))
	y_train[count] = targets	
	count = count + 1    
	if count%100 == 0:
		count = 0
		x_train, y_train=shuffle(x_train, y_train)
		trainfeatures[ti:ti+90] = x_train[:90]
		trainlabels[ti:ti+90] = y_train[:90]
		ti = ti + 90
		valfeatures[vi:vi+10] = x_train[90:]
		vallabels[vi:vi+10] = y_train[90:]
		vi = vi + 10

tr.close()
va.close()