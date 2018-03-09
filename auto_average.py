import os
import h5py
import math
import glob
import scipy
import random
import sklearn
import operator
import itertools
import collections
import numpy as np
import pandas as pd
from scipy import io
import networkx as nx
from sklearn import svm
from scipy import interp
from pprint import pprint
from scipy.misc import comb
from sklearn import datasets
from itertools import product
from sklearn import neighbors
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool
from matplotlib.pyplot import cm 
from sklearn import svm, datasets
from sklearn import datasets, svm
from sklearn.utils import shuffle
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.optimize import leastsq
from collections import defaultdict
from itertools import product, cycle
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from matplotlib import pyplot, patches
from matplotlib.colors import Normalize
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import samples_generator
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc

fc7_labels = pd.read_csv('final_matched_labels.csv', sep=",", header=None)
print('\nLabels of fc7')
print(fc7_labels.shape)
fc7_list = pd.read_csv('auto_list.txt', sep=" ", header=None)
print('\nList of sequence of fc7')
print(fc7_list.shape)
fc7_list_names = fc7_list.drop(fc7_list.columns[[1]],1)
print('\nNames of fc7')
print(fc7_list_names.shape)
fc7_list_names.to_csv('fc7_list_names.txt',sep=' ', index=False, header=False)
fc7 = h5py.File('auto_understanding_fc7.hdf5', 'r')
fc7_dataset = fc7.get('output_4096')
print(fc7.keys())
fc7_array = fc7['output_4096'][()]
fc7_array = fc7_array.T
print('\nShape of fc7 features of ads : ')
print(fc7_array.shape)
#sed -e 's/^\|$/"/g' fc7_list_names.txt > fc7_quotes.txt
fc7_quotes = pd.read_csv('fc7_quotes.txt', sep=" ", header=None)

with open('fc7_quotes.txt') as infile:
    counts = collections.Counter(l.strip() for l in infile)

fc7_average_values = list(counts.values())

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
f_line = 22
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
print('\nChecking if each line has 4096 dimensions')
print(len(final_averages[random.randint(0,950)]))

all_data = np.array(final_averages[0:949])
all_labels = fc7_labels
video_labels = all_labels[0].as_matrix()
effective_labels = all_labels[1].as_matrix()
exciting_labels = all_labels[2].as_matrix()
funny_labels = all_labels[3].as_matrix()
language_labels = all_labels[4].as_matrix()
sentiments_labels = all_labels[5].as_matrix()
topics_labels = all_labels[6].as_matrix()
path_labels = all_labels[7].as_matrix()

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('report.csv', index = False)
#report = classification_report(y_true, y_pred, target_names=target_names)
#classifaction_report_csv(report)

print("\nPredicting funny(0,1) for video ads using binary classifier:\n")
X = all_data 
funny = funny_labels
funny[funny>0.6] = 1
funny[funny<0.7] = 0
funny = funny.astype('int')
print("\nPredicting a 1v1 classifier for funny :\n")
svm_funny_1v1 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, funny).predict(X)
print(svm_funny_1v1)
print("\nPredicting a 1vrest classifier for funny :\n")
svm_funny_1vr = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, funny).predict(X)
print(svm_funny_1vr)
print("\nGetting results for 1v1 classifier using decision_function :\n")
funny_train = X[0:600]
funny_test = X[601:949]
funny_train_labels = funny[0:600]
funny_true = funny[601:949]
funny_pred = OneVsOneClassifier(LinearSVC(random_state=0)).fit(funny_train, funny_train_labels).decision_function(funny_test)
funny_pred = funny_pred.astype(int)
funny_pred[funny_pred<0] = 0
funny_pred[funny_pred>1] = 1
funny_classes = ['class funny', 'class not_funny']
print("\nClassification report for funny 1v1 :\n")
print(classification_report(funny_true, funny_pred, target_names=funny_classes))
print("\nAccuracy report for funny 1v1 :\n")
print(accuracy_score(funny_true, funny_pred))
print("\nGetting results for 1vrest classifier using decision_function :\n")
funny_train = X[0:600]
funny_test = X[601:949]
funny_train_labels = funny[0:600]
funny_true = funny[601:949]
funny_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(funny_train, funny_train_labels).decision_function(funny_test)
funny_pred = funny_pred.astype(int)
funny_pred[funny_pred<0] = 0
funny_pred[funny_pred>1] = 1
funny_classes = ['class funny', 'class not_funny']
print("\nClassification report for funny 1vr :\n")
print(classification_report(funny_true, funny_pred, target_names=funny_classes))
print("\nAccuracy report for funny 1vr :\n")
print(accuracy_score(funny_true, funny_pred))
print("\nGetting probability matrix for all examples for funny :\n")
svm_funny_fit = SVC(kernel='linear', probability=True).fit(X, funny)
svm_funny_probs = svm_funny_fit.predict_proba(X)
print(svm_funny_probs)

print("\nPredicting exciting(0,1) for video ads using binary classifier:\n")
X = all_data 
exciting = exciting_labels
exciting[exciting>0.6] = 1
exciting[exciting<0.7] = 0
exciting = exciting.astype('int')
print("\nPredicting a 1v1 classifier for exciting :\n")
svm_exciting_1v1 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, exciting).predict(X)
print(svm_exciting_1v1)
print("\nPredicting a 1vrest classifier for exciting :\n")
svm_exciting_1vr = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, exciting).predict(X)
print(svm_exciting_1vr)
print("\nGetting results for 1vrest classifier using decision_function :\n")
exciting_train = X[0:600]
exciting_test = X[601:949]
exciting_train_labels = exciting[0:600]
exciting_true = exciting[601:949]
exciting_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(exciting_train, exciting_train_labels).decision_function(exciting_test)
exciting_pred = exciting_pred.astype(int)
exciting_pred[exciting_pred<0] = 0
exciting_pred[exciting_pred>1] = 1
exciting_classes = ['class exciting', 'class not_exciting']
print("\nClassification report for exciting :\n")
print(classification_report(exciting_true, exciting_pred, target_names=exciting_classes))
print("\nAccuracy report for exciting :\n")
print(accuracy_score(exciting_true, exciting_pred))
print("\nGetting probability matrix for all examples for exciting :\n")
svm_exciting_fit = SVC(kernel='linear', probability=True).fit(X, exciting)
svm_exciting_probs = svm_exciting_fit.predict_proba(X)
print(svm_exciting_probs)

print("\nPredicting topics(38) for video ads using multiclass classifier:\n")
X = all_data 
topic = topics_labels
topic[topic<1]=1
topic = topic.astype('int')
topic_train = X[0:600]
topic_test = X[601:949]
topic_train_labels = topic[0:600]
topic_true = topic[601:949]
topic_pred = OneVsOneClassifier(LinearSVC(random_state=0)).fit(topic_train, topic_train_labels).decision_function(topic_test)
topic_pred = topic_pred.astype(int)
topic_classes = list(range(1,37))
print("\nPredicting using a multiclass classifier for topics :\n")
print("\nClassification report for topics :\n")
print(classification_report(topic_true, topic_pred, target_names=topic_classes))
print("\nAccuracy report for topics :\n")
print(accuracy_score(topic_true, topic_pred))
print("\nGetting probability matrix for all examples for topics :\n")
svm_topic_fit = SVC(kernel='linear', probability=True).fit(X, topic)
svm_topic_probs = svm_topic_fit.predict_proba(X)
print(svm_topic_probs)

print("\nPredicting sentiments(30) for video ads using multiclass classifier:\n")
X = all_data 
sentiment = sentiments_labels
sentiment[sentiment<1]=1
sentiment = sentiment.astype('int')
sentiment_train = X[0:600]
sentiment_test = X[601:949]
sentiment_train_labels = sentiment[0:600]
sentiment_true = sentiment[601:949]
sentiment_pred = OneVsOneClassifier(LinearSVC(random_state=0)).fit(sentiment_train, sentiment_train_labels).decision_function(sentiment_test)
sentiment_pred = sentiment_pred.astype(int)
sentiment_classes = list(range(1,31))
print("\nPredicting using a multiclass classifier for sentiments :\n")
print("\nClassification report for sentiments :\n")
print(classification_report(sentiment_true, sentiment_pred, target_names=sentiment_classes))
print("\nAccuracy report for sentiments :\n")
print(accuracy_score(sentiment_true, sentiment_pred))
print("\nGetting probability matrix for all examples for sentiments :\n")
svm_sentiment_fit = SVC(kernel='linear', probability=True).fit(X, sentiment)
svm_sentiment_probs = svm_sentiment_fit.predict_proba(X)
print(svm_sentiment_probs)


