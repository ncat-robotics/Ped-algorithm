# import packages
import numpy as np
import cv2
import imutils
import argparse
from skimage.filters import threshold_local
from transformations import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import pathlib
from helper_functions import run_odt_and_draw_results
from helper_functions import preprocess_image
from helper_functions import detect_objects
import help
import config
import os
 
import tensorflow as tf
from PIL import Image
import config
 
#from pyecharts.charts import Bar
 
 
#help_funcs = help.help([1200,2500])
help_funcs = help.help([476,569])
 
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
def click_event(event, x, y, flags, params):
 
   # checking for left mouse clicks
   if event == cv2.EVENT_LBUTTONDOWN:
 
       # displaying the coordinates
       # on the Shell
       print(x, ' ', y)
 
       # displaying the coordinates
       # on the image window
       font = cv2.FONT_HERSHEY_SIMPLEX
       cv2.putText(image, str(x) + ',' +
                   str(y), (x,y), font,
                   1, (255, 0, 0), 2)
       cv2.imshow('image', image)
 
   # checking for right mouse clicks  
   if event==cv2.EVENT_RBUTTONDOWN:
 
       # displaying the coordinates
       # on the Shell
       print(x, ' ', y)
 
       # displaying the coordinates
       # on the image window
       font = cv2.FONT_HERSHEY_SIMPLEX
       b = image[y, x, 0]
       g = image[y, x, 1]
       r = image[y, x, 2]
       cv2.putText(image, str(b) + ',' +
                   str(g) + ',' + str(r),
                   (x,y), font, 1,
                   (255, 255, 0), 2)
       cv2.imshow('image', image)
 
 
# construct argument parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default='/home/autodrone-hardware/Documents/GitHub/Sensor_Sandbox/Out.jpg')
args = vars(ap.parse_args())
 
# load image, compute the ratio of old height
# to the new height, clone and resize
image = cv2.imread('/home/autodrone-hardware/Documents/GitHub/Sensor_Sandbox/20221026_221921_jpg.rf.393d11b2d4d25bd4680dcc8f6a94cc15.jpg')
"""
ratio = image.shape[0] / 500.0
original = image.copy()   
image = imutils.resize(image, height=500)
 print(str(ro
node = []
 
# convert image to gray, blur and apply edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(image, 150, 200)
 
# show original image and edge detected image
print('applying edge detection')
cv2.imshow('Original Image', image)
cv2.imshow('Edged', edged)
 
# find contours and keep largest one
# initialize screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
 
 
# loop through contours
for c in cnts:
   # approximate contours
   peri = cv2.arcLength(c, True)
   approx = cv2.approxPolyDP(c, 0.02*peri, True)
 
   # check if approximated contour has four points
   if len(approx) == 4:
       screen_contours = approx
       break
   else:
       print('Contours not detected')
       break
 
# show contour outline of document
print('finding contours of document')
cv2.drawContours(image, [screen_contours], -1, (255,0,0), 2)
image = imutils.resize(image, width=image.shape[0])
cv2.imshow('Document Outline', image)
 
 
# apply transform
warped = perspective_transform(original, screen_contours.reshape(4,2)* ratio)
 
# convert to grayscale then threshold it
#warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#T = threshold_local(warped, 11, offset=10, method='gaussian')
#warped = (warped>T).astype('uint8') * 255
 
# show original image and scanned images
print('applying perspective transform')
 
 
 
name = args['image'][:len(args['image'])-4]
filename = name+'_scanned.jpg'
cv2.imwrite(filename, warped)
print('saved scan: {}'.format(filename))
"""

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
 
 
"""
detection_result_image = help_funcs.run_odt_and_draw_results_2(import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist
import config
import math

CLASSES = config.CLASSES

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
values = []
   values.append(self.)
def preprocess_image_2(image_path, input_size):
  ''' Preprocess the input image to feed to the TFLite model
  '''
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image

/home/dezmon/Documents/document_scanner/result/ouput.jpg
def detect_objects_2(interpreter, image, threshold):
  ''' Returns a list of detection results, each a dictionary of object info.
  '''

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results_2(image_path, interpreter, threshold=0.5, node=[]):
  ''' Run object detection on the input image and draw the detection results
  '''
  
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image_2(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects_2(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    


    # Find the class index of the current object
    class_id = int(obj['class_id'])
    print(class_id)
    x_ratio = 0.0085
    y_ratio = 0.0096
    start_x = 1200 
    start_y = 2500 

    real_start_x = start_x * x_ratio
    real_start_y = 24.13
    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(CLASSES[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    (mX, mY) = midpoint((xmin, ymin), (xmax, ymax))
    real_mX = mX * x_ratio
    real_mY = mY * y_ratio
    cv2.circle(original_image_np, (start_x,start_y), 5, color, -1)
    cv2.circle(original_image_np, (int(mX), int(mY)), 5, color, -1)
    cv2.line(original_image_np, (start_x,start_y), (int(mX), int(mY)),
			color, 2)
    
    D =  math.sqrt((pow((real_start_x-name = args['image'][:len(args['image'])-4]
    filename = name+'_scanned.jpg'
    cv2.imwrite(filename, warped)
    print('saved scan: {}'.format(filename))real_mX),2) + pow((real_start_y- real_mY),2)))
    print(D)
    if(class_id == 0 or class_id == 7 or class_id == 8 or class_id == 9 or class_id == 10 ):
      node.append(-D)
    elif(class_id == 3 or class_id == 4 or class_id == 5 or class_id == 6 ):
      node.append(D)


    
    
    cv2.putText(original_image_np, "{:.1f}cm".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8print(groups)
)
 
img = Image.fromarray(detection_result_image)
img.save(f'{cwd}/result/ouput2.jpg')
print('-'*100)
print('See the result folder.')
 
 
image = cv2.imread('/home/dezmon/Documents/document_scanner/result/Out_scanned.jpg')
image = imutils.resize(image,height=100,width=100)
cv2.imshow('Scanned Document2', image)
   # setting mouse handler for the image
   # and calling the click_event() functio
"""
vals = help_funcs.return_tree()
 
"""
bar = Bar()
bar.add_xaxis(["p0", "p1", "p2", "p3", "p4", "p5", "p6"])
bar.add_yaxis("商家A", vals)
bar.add_yaxis("商家B", [57, 134, 137, 129, 145, 60, 49])
bar.render()  # generate a local HTML file
"""
 
cv2.waitKey(0)
cv2.destroyAllWindows()

