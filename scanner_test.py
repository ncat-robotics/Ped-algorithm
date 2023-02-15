# import packages
import numpy as np
import cv2
import imutils
import argparse 
from skimage.filters import threshold_local
from transformations import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import numpy as np


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

model = core.Model.load('model_weights.pth', ['Duck', 'Red Chips', 'Green Chips','Green Pedestal','Green Pedestal_Tilt','Red Pedestal','Red Pedestal_Tilt','Duck_Left','Duck_Right','Duck_Front','Duck_Back'])


# construct argument parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default=r'C:\Users\estde\Documents\Object Detection\Out.jpg')
args = vars(ap.parse_args())

# load image, compute the ratio of old height 
# to the new height, clone and resize
image = cv2.imread(r'C:\Users\estde\Documents\Object Detection\20221101_182957.jpg')
ratio = image.shape[0] / 500.0
original = image.copy()    
image = imutils.resize(image, height=500)

# convert image to gray, blur and apply edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(image, 150, 200)

# show original image and edge detected image 
print('applying edge detection')
cv2.imshow('Original Image', image)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
#cv2.destroyAllWindows()

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
#cv2.waitKey(0)
#cv2.destroyAllWindows()

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

image2 = cv2.imread(r'C:\Users\estde\Documents\Object Detection\Out_scanned.jpg')

predictions = model.predict(image2)
labels, boxes, scores = predictions
thresh=0.5
filtered_indices=np.where(scores>thresh)
filtered_scores=scores[filtered_indices]
filtered_boxes=boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
newwarp = show_labeled_image(image2,filtered_boxes,filtered_labels)

objs = []
for i in  filtered_boxes:
        print( "i = ")
        print(i)
        j = 0
        t = 0
        index1 = 0
        labelstr = ''
        while j < 4:
            if(j == 0):
                x = i[j]
                print("yep1")
            if(j == 1):
                y = i[j]
                print("yep2")
            if(j == 2):
                w = i[j]
                print("yep3")
                
            if(j == 3):
                h = i[j]
                print("yep4")
            print( "j = ")
            print(j)
            j = j + 1
        objs.append(midpoint(x,y,w,h))
        print(objs[0])
#plt.plot((0,0),objs[0])
img = plt.imread(filename)
plt.imshow(img)
cv2.imshow('Original', imutils.resize(original, height=650))
cv2.imshow('Scanned Document', imutils.resize(warped, height=650))

	# setting mouse handler for the image
	# and calling the click_event() functio
cv2.waitKey(0)
