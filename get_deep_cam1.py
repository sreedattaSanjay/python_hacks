import os
import csv
import sys
import math
import glob
import scipy
import caffe
import sklearn
import operator
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from vgg16 import VGG16
from pprint import pprint
from scipy import spatial
from sklearn import datasets
from itertools import product
from sklearn import neighbors
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.pyplot import cm 
from scipy.spatial import distance
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy import io, ndimage, misc
from sklearn.decomposition import PCA
from keras.preprocessing import image
from matplotlib import pyplot, patches
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from imagenet_utils import preprocess_input
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('VGG_ILSVRC_19_layers_deploy.prototxt','VGG_ILSVRC_19_layers.caffemodel',
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1,3,224,224)

rootdir = '/home/cs16mtech11021/graph_and_flow/graph/datasets/iLIDS-VID/i-LIDS-VID/sequences/cam1/'
idx = 0
deep_features = []
pixel_info = []
flat_arrays = []
flat_vectors = []
len_images = []
list_subdirects = []
img_list = []
dynamic_name = 'test'


def get_all_features():
	for i in range (0, len(img_list)-1):
		im = caffe.io.load_image(img_list[i].rstrip())
		net.blobs['data'].data[...] = transformer.preprocess('data', im)
		out = net.forward()
		# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
		#maps3_4 = net.blobs['conv3_4'].data[0,:]
		#maps4_4 = net.blobs['conv4_4'].data[0,:]
		#maps5_4 = net.blobs['conv5_4'].data[0,:]
		output_graph = net.blobs['fc8'].data[0,:]
		output_file = 'cam1_'+files+'_vgg16_features_'+ dynamic_name +'.txt'
		np.savetxt(output_file, net.blobs['fc8'].data[0,:], fmt='%.4f', delimiter='\n')
		return output_graph
		#print(output_graph)

for subdir, dirs, files in os.walk(rootdir):
	for files in dirs:
		print(files)
		print(os.getcwd())
		changedir = rootdir + files
		list_subdirects.append(changedir)
		variable = os.listdir(changedir)
		os.chdir(changedir)
		print(os.getcwd())
		len_images.append(len(variable))
		temp_deep_features = []
		for i in range(len(variable)):
			print(variable[i])
			dynamic_name, file_extension = os.path.splitext(variable[i])
			img_list = variable
			temp_deep = get_all_features()
			temp_deep_features.append(temp_deep)
			fln = variable[i]
		df = pd.read_csv(temp_deep_features)
		df.to_csv(fln+".csv", header=False, index=False)
		os.chdir(rootdir)
		print(os.getcwd())
os.chdir(rootdir)