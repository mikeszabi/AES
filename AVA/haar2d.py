__author__ = 'SzMike'

import sys
import cv2
import pywt
import easygui
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import math
import utils
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

IMAGE_DIR=r'c:\Users\PocUser\Documents\DATA\AES\top'

def block_view(array, block= (3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (int(array.shape[0]/ block[0]), int(array.shape[1]/ block[1]))+ block
    strides= (block[0]* array.strides[0], block[1]* array.strides[1])+ array.strides
    return ast(array, shape= shape, strides= strides)

def bbImage(img,maxBBSize,nLevel,divisional):
    scale= maxBBSize/float(max(img.shape[0],img.shape[1]))   
    if scale<1:
        img_small=cv2.resize(img, (int(scale*img.shape[1]),int(scale*img.shape[0])), interpolation = cv2.INTER_AREA)
    else:
        img_small=img

    if img_small.ndim==2:
        img_small=img_small[:,:,np.newaxis]
        
    modSize=(img_small.shape[0]%(divisional*pow(2,nLevel)),img_small.shape[1]%(divisional*pow(2,nLevel)))
    img_bb=img_small[int(math.ceil(modSize[0]/2)):img_small.shape[0]-int(math.floor(modSize[0]/2)),int(math.ceil(modSize[1]/2)):img_small.shape[1]-int(math.floor(modSize[1]/2)),:]
   
    return img_bb
            

def haarImage(img,nLevel=3):
    imgHaar=np.empty(img.shape, dtype='float32')    
    coeffs = pywt.wavedec2(img, 'haar', level=nLevel)
    for i, coef in enumerate(coeffs): 
        if (i==0):
            imgHaar[0:coef.shape[0],0:coef.shape[1]]=coef/pow(2,3);
        else:
            imgHaar[coef[0].shape[0]:2*coef[0].shape[0],0:coef[0].shape[1]]=coef[0]/pow(2,(nLevel+1-i))
            imgHaar[0:coef[1].shape[0],coef[1].shape[1]:2*coef[1].shape[1]]=coef[1]/pow(2,(nLevel+1-i))
            imgHaar[coef[2].shape[0]:2*coef[2].shape[0],coef[2].shape[1]:2*coef[2].shape[1]]=coef[2]/pow(2,(nLevel+1-i))
    return imgHaar
    
def calcHaarEmap_oneLevel(haar1):
    # 3D, max among channels
    half_height = int(haar1.shape[0]/2);
    half_width  = int(haar1.shape[1]/2);

    imax=np.empty((half_height,half_width,3), dtype='float32') 

    imax[:,:,0]=abs(haar1[0:half_height,half_width:2*half_width,]).max(2) # horizontal
    imax[:,:,1]=abs(haar1[half_height:2*half_height,0:half_width,:]).max(2) # vertical
    imax[:,:,2]=abs(haar1[half_height:2*half_height,half_width:2*half_width,:]).max(2) # diagonal

    emap=abs(imax).max(2) # max over blocks
    return emap

def calcHaarEmap(haar,nLevel=3):
    (h, w, c) = haar.shape
    sh = int(h/(pow(2,nLevel)))
    sw = int(w/(pow(2,nLevel)))
    eMaps=np.empty((sh,sw,nLevel), dtype='float32')    

    for iLevel in range(0,nLevel):
         sh = int(h/pow(2,iLevel))
         sw = int(w/pow(2,iLevel))
         tmp=calcHaarEmap_oneLevel(haar[0:sh,0:sw,:])
         eMaps[:,:,nLevel-iLevel-1]=block_view(tmp,
                                    block=(int(pow(2,(nLevel-iLevel-1))),int(pow(2,(nLevel-iLevel-1))))).max(2).max(2)
    
    return eMaps

def calcFocusMask(eMaps,edgeTsh):

    focusMask=np.empty(eMaps.shape, dtype='bool')    
    focusMask[:,:,0]=eMaps.min(2)>edgeTsh
    for iLevel in range(1,eMaps.shape[2]):
         focusMask[:,:,iLevel]= np.bitwise_and((eMaps[:,:,iLevel] >  eMaps[:,:,iLevel-1]),focusMask[:,:,iLevel-1]==1)
         
    return focusMask
    
def doMorphology(mask):
    r=int(max(mask.shape)/20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask
    
def dominantColor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = 5)
    clt.fit(image)
    hist = utils.centroid_histogram(clt)
    bar = utils.plot_colors(hist, clt.cluster_centers_)
 
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    
def underOver(image):
    img_tmp=np.empty(image.shape, dtype='uint8')   
    img_tmp.fill(0)
    image_gray = cv2.cvtColor(image.astype(dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    mask_o=np.empty(image_gray.shape, dtype='bool') 
    mask_o=image_gray==255
    mask_o=255*mask_o.astype(dtype=np.uint8)
    mask_o=doMorphology(mask_o)
    img_tmp[:,:,2]=mask_o
    mask_u=np.empty(image_gray.shape, dtype='bool') 
    mask_u=image_gray==0
    mask_u=255*mask_u.astype(dtype=np.uint8)
    mask_u=doMorphology(mask_u)
    img_tmp[:,:,0]=mask_u
    img_uo = cv2.addWeighted(image.astype(dtype=np.uint8),0.5,img_tmp,0.5,0)

    return img_uo

def main(photo_file):

    isGray=0   
    divisional=3.0
    nLevel=3
    residual=64.0
    edgeTsh=5
    maxBBSize=residual*pow(2,nLevel)*divisional;
    
    img_orig = cv2.imread(photo_file)

    if isGray:
        img_gray = cv2.cvtColor(img_orig.astype(dtype=np.uint8), cv2.COLOR_BGR2GRAY)

        img=np.stack((img_gray,)*3, axis=-1)
    else:
        img=img_orig
        
    #cv2.imshow('gray',image_gray) #.astype(dtype=np.uint8))
    #cv2.waitKey()

    img_bb=bbImage(img,maxBBSize,nLevel,divisional)   
    
    if isGray:
        img_bb3=np.repeat(img_bb, 3, axis=2)
    else:
        img_bb3=img_bb
        
    haar=np.empty(img_bb3.shape, dtype='float32')    

    for iCh in range(img_bb3.shape[2])  :  
        haar[:,:,iCh]=haarImage(img_bb3.astype(np.float32)[:,:,iCh],nLevel)
    
    eMaps=calcHaarEmap(haar,nLevel)
    
    focusMask=calcFocusMask(eMaps,edgeTsh).astype('uint8')
    fMask=doMorphology(focusMask[:,:,2])

    cv2.imshow('fMask',255*focusMask[:,:,0]) #.astype(dtype=np.uint8))
    cv2.waitKey(1)
    
    img_tmp=np.empty(focusMask.shape, dtype='uint8')   
    img_tmp.fill(0)
    img_tmp[:,:,0]=fMask
    cv2.normalize(img_tmp,img_tmp,255,0,cv2.NORM_MINMAX)
    
    img_foc = cv2.addWeighted(img_bb3,0.5,cv2.resize(img_tmp,(img_bb.shape[1],img_bb.shape[0])),0.5,0)    
        
    (h, w, c) = haar.shape
    sh = int(h/(pow(2,nLevel)))
    sw = int(w/(pow(2,nLevel)))
    img_small=haar[0:sh,0:sw,:]
    
    dominantColor(img_small)
    
    img_uo=underOver(img_bb3.astype(dtype=np.uint8))

    cv2.imshow('fmap',cv2.resize(img_foc,(600,int((600/float(img_foc.shape[1])*img_foc.shape[0]))))) #.astype(dtype=np.uint8))
    cv2.waitKey(1)
    cv2.imshow('underover',cv2.resize(img_uo,(600,int((600/float(img_uo.shape[1])*img_uo.shape[0]))))) #.astype(dtype=np.uint8))
    cv2.waitKey(1)
    
    cv2.imwrite(r'c:\Users\PocUser\Documents\DATA\TEMP'+'foc_'+os.path.split(photo_file)[1],img_foc)
    cv2.imwrite(r'c:\Users\PocUser\Documents\DATA\TEMP'+'uo_'+os.path.split(photo_file)[1],img_uo)

if __name__ == '__main__':
    msg = 'Do you want to continue?'
    title = 'Please Confirm'
    photo_file=os.path.join(IMAGE_DIR,'nice_30.jpg')
    #main(photo_file)
    while easygui.ccbox(msg, title):     # show a Continue/Cancel dialog
        photo_file = easygui.fileopenbox(default=photo_file)
        cv2.destroyAllWindows() # Ok, destroy the window
        main(photo_file)
    sys.exit(0)