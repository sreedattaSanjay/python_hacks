import time
start = time.clock()

import os
import csv
import cv2
import math
import glob
import scipy
import sklearn
import operator
import itertools
import numpy as np
import pandas as pd
from scipy import io
import networkx as nx
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
from sklearn.decomposition import PCA
from matplotlib import pyplot, patches
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

fc7_array = pd.read_csv('cam1_combined_deep_features.csv', sep=" ", header=None)

fc7_quotes = pd.read_csv('cam1_num_images.csv', sep=" ", header=None)
fc7_average_values = fc7_quotes[0]

def averages(matrix):
	matrix = matrix.T
	avg_matrix = [sum(row)/len(row) for row in matrix]
	#print(len(avg_matrix))
	return avg_matrix

test_average_2 = []
def averages_2(matrix):
	for i in range(0, len(matrix)):
		matrix_2 = matrix
		test_average_1 = np.mean(matrix[i])
		return test_average_1
		#test_average_2.append(test_average_1)

def write_list_to_file(guest_list, filename):
    """Write the list to csv file."""
    with open(filename, "w") as outfile:
        for entries in guest_list:
            outfile.write(str(entries))
            outfile.write(" ")
        outfile.write(",")
        outfile.write("\n")

final_averages = []
s_line = 0
f_line = 5
printline = 0
fc7_array_transpose = fc7_array.T

for a_value in fc7_average_values:
	chunk = fc7_array_transpose[s_line:f_line]
	print("chunk shape : ")
	print(chunk.shape)
	average_chunk = averages(chunk)
	print("average_chunk length which should be 4096 : ")
	print(len(average_chunk))
	final_averages.append(average_chunk)
	#final_averages.append("\n")
	print("final_averages which should be increasing +1 : ")
	print(len(final_averages))
	s_line = s_line + a_value
	f_line = a_value + f_line
	print("Features done : ", printline)
	printline = printline + 1

print(len(final_averages))
write_list_to_file(final_averages, "final_averages.csv")
print('\nLength of final_averages')
print(len(final_averages))

#Drawing a graph from gallery features by finding adjacency matrix by using knn
X = gallery_features.T
y = gallery_outcomes.T
print("\nGallery_features before normalization: ")
print(len(X))
print("\nShape of gallery_features before normalization: ")
print(X.shape)
standardizer = StandardScaler()
X_std = X#standardizer.fit_transform(X)
print("\nGallery_features after normalization: ")
print(len(X_std))
print("\nShape of gallery_features after normalization: ")
print(X_std.shape)


#Drawing a graph from gallery features by finding adjacency matrix by using knn
M = probe_features.T
n = probe_outcomes.T
print("\nProbe_features before normalization: ")
print(len(M))
print("\nShape of probe_features before normalization: ")
print(M.shape)
standardizer = StandardScaler()
M_std = M #standardizer.fit_transform(M)
print("\nProbe_features after normalization: ")
print(len(M_std))
print("\nShape of probe_features after normalization: ")
print(M)#_std.shape)

print("\nCreating each graph for each node i.e person from gallery images")
print("\nLoading deep features of gallery images after normalization")
g_f = X_std
print(g_f.shape)
g_y = y.values.ravel()
#print(g_y)
print("Done")
print("\nLoading deep features of probe images after normalization")
p_f = M_std
print(p_f.shape)
p_y = n.values.ravel()
#print(p_y)
print("Done\n")
K_value = 101

def GaussianMatrix(X,sigma):
    row,col=X.shape
    GassMatrix=np.zeros(shape=(row,row))
    X=np.asarray(X)
    i=0
    for v_i in X:
        j=0
        for v_j in X:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            j+=1
        i+=1
    return GassMatrix

def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))
    
g_f_mat = X.as_matrix()
g_f_mat = g_f_mat.T
p_f_mat = M.as_matrix()
p_f_mat = p_f_mat.T
g_nearest_100_neighbs = []
p_nearest_100_neighbs = []
g_newSamples = []
p_newSamples = []
g_tree = spatial.KDTree(g_f_mat, leafsize = g_f_mat.shape[0]+1)

for i in range(0, 2): #len(g_f_mat)
    distance, index = g_tree.query([g_f_mat[i]], k = K_value)
    g_nearest_100_neighbs.append(g_f_mat[index])
    g_temp_sample = (g_f_mat[index])*(g_f_mat[index].T)
    g_mean = np.mean(g_temp_sample, axis=0)
    g_std = np.std(g_temp_sample, axis=0)
    Samples = g_temp_sample
    c = np.hstack(Samples) 
    mean, std = np.mean(c), np.std(c)
    g_newSamples.append(np.asarray([(np.array(xi)-mean)/std for xi in Samples]))

p_tree = spatial.KDTree(p_f_mat, leafsize = p_f_mat.shape[0]+1)  

for j in range(0, 2): #len(p_f_mat)
    
    distance, index = p_tree.query([p_f_mat[j]], k = K_value)
    p_nearest_100_neighbs.append(p_f_mat[index])
    p_temp_sample = (p_f_mat[index])*(p_f_mat[index].T)
    p_mean = np.mean(p_temp_sample, axis=0)
    p_std = np.std(p_temp_sample, axis=0)
    Samples = p_temp_sample
    c = np.hstack(Samples) 
    mean, std = np.mean(c), np.std(c)
    p_newSamples.append(np.asarray([(np.array(xi)-mean)/std for xi in Samples]))
print(g_nearest_100_neighbs[0].shape)    
print(g_temp_sample.shape)
print(g_newSamples[0].shape )

 # g_nn_euclidean1 = NearestNeighbors(n_neighbors=100, metric='euclidean').fit(X_std1, y1)
 #    g_neighbors_graph1 = g_nn_euclidean1.kneighbors_graph(X_std1)   
 #    g_nearest_neighbors_with_self_array1 = g_nn_euclidean1.kneighbors_graph(X_std1).toarray()
 #    for i, x in enumerate(g_nearest_neighbors_with_self_array1):
 #        x[i] = 0
 #    g_Z1 = nx.Graph(g_nearest_neighbors_with_self_array1)


stop = time.clock()
print("\nTotal time taken (in seconds) to create graphs from adjacency matrix for both gallery and probe images: ")
print(stop-start)

