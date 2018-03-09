import os
import csv
import sys
import glob
import pandas as pd

rootdir = '/home/cs16mtech11021/graph_and_flow/graph/datasets/iLIDS-VID/i-LIDS-VID/sequences/cam2/'

idx = 0
deep_features = []
len_images = []
list_subdirects = []
combine_csv = []
k = 0

for subdir, dirs, files in os.walk(rootdir):
	for files in dirs:
		print(files)
		print(os.getcwd())
		changedir = rootdir + files
		list_subdirects.append(changedir)
		variable = os.listdir(changedir)
		os.chdir(changedir)
		print(os.getcwd())
		len_images.append(len(glob.glob("*.txt")))
		for i in glob.glob("*.txt"):
			fln = i
			print(fln)
			df = pd.read_csv(fln,delim_whitespace=True, header=None)
			df = df.transpose()
			df = df.values.tolist()
			combine_csv.append(df[0])
		os.chdir(rootdir)
		print(os.getcwd())
os.chdir(rootdir)

with open('combined_deep_features.csv', 'w') as myfile1:
    wr1 = csv.writer(myfile1)
    for row in combine_csv:
    	wr1.writerow(row)

with open('combined_deep_features_2.csv', 'w') as myfile2:
    wr2 = csv.writer(myfile2, delimiter=',')
    wr2.writerow(combine_csv)

with open('num_images.csv', "w") as myfile3:
    wr3 = csv.writer(myfile3, delimiter=',')
    wr3.writerows([[lens] for lens in len_images])

with open('subdirectories.csv', "w") as myfile4:
    wr4 = csv.writer(myfile4, delimiter=',')
    wr4.writerows([[lsubs] for lsubs in list_subdirects])
