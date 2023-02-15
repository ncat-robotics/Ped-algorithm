# Python code for Multiple Color Detection


import numpy as np
import cv2
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import argparse
from PIL import Image
"""
import rospy
from std_msgs.msg import String
   
def LEDStart():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
    hello_str = "LEDON %s" % rospy.get_time()
    rospy.loginfo(hello_str)
    pub.publish(hello_str)
    rate.sleep()
   
if __name__ == '__main__':
    try:
        LEDStart()
    except rospy.ROSInterruptException:
        pass

"""



# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while(1):
	
        # Reading the video from thewhile not rospy.is_shutdown():
        # webcam in image frames
        _, imageFrame = webcam.read()

        #hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # threshold the image to reveal light regions in the
        # blurred image
        thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]

        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # perform a connected component analysis on the thresholded
        # image, then initialize a mask to store only the "large"
        # components
        labels = measure.label(thresh, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels 
            ####Send Meassage#####
            #LEDStart()
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # if the number of pixels in the component is esufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 300:
                mask = cv2.add(mask, labelMask)
        # find the contours in the mask, then sort them from left to
        # right
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(imageFrame, (int(cX), int(cY)), int(radius),
                (0, 0, 255), 3)
            cv2.putText(imageFrame, "#{}".format(i + 1), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # show the output image
        cv2.imshow("Image", imageFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
webcam.release()
cv2.destroyAllWindows()



	
