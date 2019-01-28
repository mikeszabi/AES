# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:27:36 2017

@author: SzMike
"""

#!/usr/bin/env python
# coding: utf8

import cv2

import numpy as np
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
    inScaleFactor = 1
    meanVal = 127.5

    
    def __init__(self):
        # AVA
        DEPLOY = 'model\initModel.prototxt'
        MODEL_FILE = 'model\initModel.caffemodel'
        #Size of images
        self.net = cv2.dnn.readNetFromCaffe(DEPLOY, MODEL_FILE)
        
    def get_scores(self,image_files):      

        im_all_scores=[None] * len(image_files)
        for i, fname in enumerate(image_files):
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            if (type(img) is np.ndarray):
                print(fname)
                img = transform_img(img, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
            
                blob=cv2.dnn.blobFromImage(img, self.inScaleFactor, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), self.meanVal)
                self.net.setInput(blob)
                out = self.net.forward()

                print(str(i))
            
                im_all_scores[i] = {'AestheticScore':str(out[0][0])}
            else:
                print(fname+' does not exists')
                im_all_scores[i] = {'AestheticScore':str(None)}
        
        return im_all_scores
        


