# Load training and test images for one-shot classification of omniglot dataset.
# Code inspired by https://github.com/brendenlake/omniglot/blob/master/python/one-shot-classification/demo_classification.py 

from __future__ import print_function

import os
import copy
import numpy as np
from scipy.ndimage import imread
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

def expand_images(img_set, out_size):
	new_imset = []
	for im in img_set:
		im = im.reshape(105, 105)
		nz = np.nonzero(im)
		nzcol = np.sort(nz[0])
		nzrow = np.sort(nz[1])

		new_imset.append(resize(im[nzcol[0]:nzcol[-1]+1, nzrow[0]:nzrow[-1]+1], out_size))
	new_imset = np.array(new_imset)
	return new_imset.reshape(new_imset.shape + (1,))


class Dataset:
	"""
	Easy access to data for omniglot runs.
	"""
	def __init__(self, train_X, train_y, test_X, test_y):

		self.train_X = train_X
		self.train_X_expanded = expand_images(train_X, (105, 105))
		self.train_X_small = expand_images(train_X, (34, 34))
		self.train_y = train_y
		self.train_y_cat = to_categorical(train_y, num_classes=20)

		self.test_X = test_X
		self.test_X_expanded = expand_images(test_X, (105, 105))
		self.test_X_small = expand_images(test_X, (34, 34))
		self.test_y = test_y
		self.test_y_cat = to_categorical(test_y, num_classes=20)

def LoadImgAsPoints(fn):
	I = imread(fn,flatten=True)
	I = np.array(I,dtype=bool)
	I = np.logical_not(I)

	return np.array(I, dtype=int).reshape(I.shape + (1,))

def load_images(base_dir):
	all_runs = []

	for r in range(1, 21):
		rs = str(r)
		if len(rs) == 1:
			rs = '0' + rs		
		with open(base_dir+'/'+'run'+rs+'/class_labels.txt') as f:
			content = f.read().splitlines()

		pairs = [line.split() for line in content]
		test_files  = [pair[0] for pair in pairs]
		train_files = [pair[1] for pair in pairs]

		test_y  = np.array([int(a.split('/')[-1].split('.')[0].replace('class', '')) - 1 for a in train_files])

		train_files.sort()
		train_y = np.array([int(a.split('/')[-1].split('.')[0].replace('class', '')) - 1 for a in train_files])

		train_items = np.array([LoadImgAsPoints(base_dir+'/'+f) for f in train_files])
		test_items  = np.array([LoadImgAsPoints(base_dir+'/'+f) for f in test_files ])

		all_runs.append(Dataset(train_items, train_y, test_items, test_y))

	return all_runs