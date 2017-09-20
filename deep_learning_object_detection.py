# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
#import argparse
import cv2

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
%matplotlib qt5
from file_helper import imagelist_in_depth

print(cv2.__version__)
prototxt='model\initModel.prototxt'
model='model\initModel.caffemodel'
image_file=r'.\\images\\t2plY.jpg'

IMAGE_MEAN= 'model\mean_AADB_regression_warp256.binaryproto'

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
net.getLayerNames()
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

im_files=imagelist_in_depth(r'c:\Users\szmike\Documents\DATA\studio2')

im_file=im_files[1]
image = cv2.imread(im_file)
print(im_file)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (127.5,127.5,127))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward('fc11_score')
print(detections)


fig = plt.figure(1, (4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(int(np.ceil(np.sqrt(len(im_files)))), int(np.ceil(np.sqrt(len(im_files))))),  
                 axes_pad=0.1,  # pad between axes in inch.
                 )
for i in range(len(im_files)):
    im_file=im_files[i]
    image = cv2.imread(im_file)
    print(im_file)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (127.5,127.5,127))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward('fc11_score')
    print(detections)
    grid[i].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))  # The AxesGrid object work as a list of axes.
    grid[i].text(0.1,0.1,str(detections[0][0]),color='green',fontsize=15)
plt.show()


# show the output image
small_image=cv2.resize(image,(227,227))
cv2.imshow("Output", small_image)
cv2.putText(small_image, str(detections), (10, 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2, 2)
cv2.waitKey(0)