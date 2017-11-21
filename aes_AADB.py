# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:27:36 2017

@author: SzMike
"""

#!/usr/bin/env python
# coding: utf8

# make sure that caffe is on the python path
import sys
import caffe

import os
import glob
import cv2
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from file_helper import imagelist_in_depth



#   Image processing helper function
def transform_img(img, img_width=227, img_height=227):
#   Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

class scoring:

#Size of images
    
    IMAGE_WIDTH = 227
    IMAGE_HEIGHT = 227
    
    def __init__(self):
        # AVA
        IMAGE_MEAN= r'.\\model\\mean_AADB_regression_warp256.binaryproto'
        DEPLOY = 'model\initModel.prototxt'
        MODEL_FILE = 'model\initModel.caffemodel'
        caffe.set_mode_cpu()
        #Size of images


# Reading mean image, caffe model and its weights

        self.input_layer = 'imgLow'
        mean_blob = caffe_pb2.BlobProto()
        with open(IMAGE_MEAN,'rb') as f:
            mean_blob.ParseFromString(f.read())
        mean_array=np.asarray(mean_blob.data, dtype=np.float32).reshape((mean_blob.height, mean_blob.width, mean_blob.channels))
        #cv2.imshow("Output", mean_array)
        #cv2.waitKey(0)    
        mean_array = transform_img(mean_array,self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        mean_array = mean_array.reshape((mean_blob.channels, self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        #Read model architecture and trained model's weights
        self.net = caffe.Net(DEPLOY,
                        MODEL_FILE,
                        caffe.TEST)

        #Define image transformers
        print('Shape mean_array : '+str(mean_array.shape))
        print('Shape net: '+str(self.net.blobs[self.input_layer].data.shape))
        self.net.blobs[self.input_layer].reshape(1,        # batch size
                                      3,         # channel
                                      self.IMAGE_WIDTH, self.IMAGE_HEIGHT)  # image size
        self.transformer = caffe.io.Transformer({self.input_layer: self.net.blobs[self.input_layer].data.shape})
        self.transformer.set_mean(self.input_layer, mean_array)
        self.transformer.set_transpose(self.input_layer, (2,0,1))
        
        
    def get_scores(self,image_files):      

        im_all_scores=[None] * len(image_files)
        for i, fname in enumerate(image_files):
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            if (type(img) is np.ndarray):
                print(fname)
                img = transform_img(img, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
            
                self.net.blobs[self.input_layer].data[...] = self.transformer.preprocess(self.input_layer, img)
                out = self.net.forward()
                
                print(str(i))
            
                im_all_scores[i] = {'AestheticScore':str(out['fc11_score'][0][0])}
            else:
                print(fname+' does not exists')
                im_all_scores[i] = {'AestheticScore':str(None)}
        
        return im_all_scores
        

