import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os
import cv2
import sys

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

img_list = open('/home/cs16mtech11021/graph/probe/probe_names.txt','r')
img_list = img_list.readlines()

#variable = os.listdir("/home/cs16mtech11021/graph/gallery/data/")
os.chdir("/home/cs16mtech11021/graph/probe/data/")

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
		output_file = 'probe_vgg16_features_%d.txt' % i
		np.savetxt(output_file, net.blobs['fc8'].data[0,:], fmt='%.4f', delimiter='\n')
		#print(output_graph)

get_all_features()