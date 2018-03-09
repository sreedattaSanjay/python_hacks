import os
import csv
import math
import glob
import scipy
import sklearn
import operator
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image
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
from matplotlib import pyplot, patches
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier



rootdir = '/home/cs16mtech11021/graph_and_flow/graph/datasets/iLIDS-VID/i-LIDS-VID/sequences/cam1/'
idx = 0
pixel_info = []
flat_arrays = []
flat_vectors = []
len_images = []
list_subdirects = []

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
		for i in range(len(variable)):
			print(variable[i])
			img = Image.open(variable[i])
			pix = list(img.getdata())
			pixel_info.append(pix)
			arr = np.array(img)
			flat_arr = arr.ravel()
			flat_arrays.append(flat_arr)
			flat_vect = np.matrix(flat_arr)
			flat_vectors.append(flat_vect)
		os.chdir(rootdir)
		print(os.getcwd())
os.chdir(rootdir)

with open('cam1_flat_arrays.csv', 'w') as myfile1:
    wr1 = csv.writer(myfile1)
    for row in flat_arrays:
    	wr1.writerow(row)

with open('cam1_pixel_info.csv', 'w') as myfile2:
    wr2 = csv.writer(myfile2, delimiter=',')
    wr2.writerow(pix)

with open('cam1_num_images.csv', "w") as myfile3:
    wr3 = csv.writer(myfile3, delimiter=',')
    wr3.writerows([[lens] for lens in len_images])

with open('cam1_subdirectories.csv', "w") as myfile4:
    wr4 = csv.writer(myfile4, delimiter=',')
    wr4.writerows([[lsubs] for lsubs in list_subdirects])

#--------------------------------------------------------
# # do something to the vector
# vector[:,::10] = 128
# # reform a numpy array of the original shape
# arr2 = np.asarray(vector).reshape(shape)
# # make a PIL image
# img2 = Image.fromarray(arr2, 'RGBA')
# img2.show()
#--------------------------------------------------------