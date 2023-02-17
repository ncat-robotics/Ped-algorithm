# import packages
import numpy as np
import cv2
import imutils
import argparse
from skimage.filters import threshold_local
from scanning.transformations import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import pathlib
from helper_functions import run_odt_and_draw_results
from helper_functions import preprocess_image
from helper_functions import detect_objects
import map_helper_functions

import os
from PIL import Image
import model_dependencies.config as config
 
#from pyecharts.charts import Bar
 
 
#help_funcs = help.help([1200,2500]) #change reference point
help_funcs = map_helper_functions.map_help([476,569])
 
MODEL_PATH = config.MODEL_PATH
MODEL_NAME = config.MODEL_NAME
cwd = os.getcwd()
 
DETECTION_THRESHOLD = 0.3
 
# Change the test file path to your test image
#INPUT_IMAGE_PATH = 'dataset/test/IMG_2347_jpeg_jpg.rf.7c71ac4b9301eb358cd4a832844dedcb.jpg'
 
 
# Load the TFLite model
model_path = f'{MODEL_PATH}/{MODEL_NAME}'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
 
 
 
 
def midpoint(ptA, ptB,ptC,ptD):
   return ((ptA + ptB) * 0.5, (ptC + ptD) * 0.5)

 
 
# construct argument parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default='/home/autodrone-hardware/Documents/GitHub/Sensor_Sandbox/Out.jpg')
args = vars(ap.parse_args())
 
# load image, compute the ratio of old height
# to the new height, clone and resize
image = cv2.imread('/home/autodrone-hardware/Documents/GitHub/Sensor_Sandbox/20221026_221921_jpg.rf.393d11b2d4d25bd4680dcc8f6a94cc15.jpg')


name = args['image'][:len(args['image'])-4]
filename = name+'_scanned.jpg'
cv2.imwrite(filename, image)
print('saved scan: {}'.format(filename))

# Run inference and draw detection result on the local copy of the original file
detection_result_image = help_funcs.run_odt_and_draw_results_2(
   '/home/autodrone-hardware/Documents/GitHub/Sensor_Sandbox/Out_scanned.jpg',
   interpreter,
   threshold=DETECTION_THRESHOLD,
)
 
img = Image.fromarray(detection_result_image)
img.save(f'{cwd}/result/ouput.jpg')
 
 
image = cv2.imread('/home/autodrone-hardware/Documents/GitHub/Sensor_Sandbox/result/ouput.jpg')
#image = imutils.resize(image,height=100,width=100)
cv2.imshow('Scanned Document', image)



help_funcs.draw_raw_graph()
 
help_funcs.new_reference([0,0])
vals = help_funcs.return_sorted()
 
 
vals = help_funcs.return_tree()
 

 
cv2.waitKey(0)
cv2.destroyAllWindows()

