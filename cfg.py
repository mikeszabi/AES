# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:20:02 2017

@author: SzMike
"""

import os

class param:
    def __init__(self,model_type=''):
        self.model_type=model_type
        self.imgSize=None
        self.image_mean=128.0
        self.node_name=None
        if  self.model_type=='ResNet_18':
            self.imgSize=224
            self.image_mean   = 128.0
            self.model_file  = 'ResNet_18.model'
            self.node_name = "OutputNodes.z"
        elif model_type=='ResNet_152':
            self.imgSize=224
            self.image_mean   = 128.0
            self.model_file  = 'ResNet_152.model'
            self.node_name = "OutputNodes.z"
        elif model_type=='AlexNetBS':
            self.imgSize=227
            self.image_mean   = 128.0
            self.model_file  = 'AlexNetBS.model'
            self.node_name = "z"
            #node_name = "z.x._._"
        elif model_type=='AlexNetBS_2nd':
            self.imgSize=227
            self.image_mean   = 128.0
            self.model_file  = 'AlexNetBS.model'
            self.node_name = "z.x._._"
        else:
            print('Unknown model')
            #return []
        self.softmaxed=False
               
    def getDirs(self,base_folder=None):
        if base_folder is None:
            base_folder=os.path.curdir
        image_list_file=os.path.join(base_folder,'input','image_list.json')
        feature_file=os.path.join(base_folder,'output','features.json')
        model_file=os.path.join(base_folder,'models','IMAGENET',self.model_file)
        return image_list_file, feature_file, model_file
    
  