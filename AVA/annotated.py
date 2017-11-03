# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:00:36 2016

@author: SzMike
"""

import numpy as np
import cv2
import csv
import glob, re
import math

def bbImage(img_orig,maxBBSize,nLevel,divisional):
    scale= maxBBSize/float(max(img_orig.shape[0],img_orig.shape[1]))   
    if scale<1:
        img_small=cv2.resize(img_orig, (int(scale*img_orig.shape[1]),int(scale*img_orig.shape[0])), interpolation = cv2.INTER_AREA)
    else:
        img_small=img_orig
    modSize=(img_small.shape[0]%(divisional*pow(2,nLevel)),img_small.shape[1]%(divisional*pow(2,nLevel)))
    img_bb=img_small[int(math.ceil(modSize[0]/2)):img_small.shape[0]-int(math.floor(modSize[0]/2)),
                     int(math.ceil(modSize[1]/2)):img_small.shape[1]-int(math.floor(modSize[1]/2)),:]
    return img_bb

reader = csv.reader(open(r'e:\Pictures\TestSets\scores20160909.csv'),delimiter=';')
data = list(reader)
row_count = len(data)

print row_count

result = {}
for idx,row in enumerate(reader):
    if row[1]!='NULL' and row[2]!='NULL':
        fname=r'e:/Pictures/TestSets/Annotator/' + row[0] + '*'
        fname=re.sub(r'\[', '[[]', fname)
        fname = re.sub(r'(?<!\[)\]', '[]]', fname)
        fnames=glob.glob(fname)
        
        #image = cv2.imread(fnames[0])
        #image_small=bbImage(image,512,3,3)
   
        #cv2.putText(image_small,row[1],(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #cv2.putText(image_small,row[2],(30,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #cv2.imshow("Image", image_small)
        result[fnames[0]] = [row[1], row[2]]           
        #cv2.waitKey(0)
        if idx>10:
            break
