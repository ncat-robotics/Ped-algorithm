import cv2
import numpy as np


# Load the video capture device
cap = cv2.VideoCapture(1)


# Loop until the user presses the 'q' key
while True:
   # Read a frame from the video capture device
   ret, frame = cap.read()


   # Convert the frame to the HSV color space
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


   # Define the lower and upper bounds for the red color in the HSV color space
   lower_red = np.array([132, 35, 213])
   upper_red = np.array([179, 255, 255])
   lower_red2 = np.array([0, 0, 249])
   upper_red2 = np.array([0, 255, 255])


   # Create a mask that only includes the red pixels in the frame
   mask1 = cv2.inRange(hsv, lower_red, upper_red)
   mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
   mask = cv2.bitwise_or(mask1, mask2)


   # Apply a series of morphological operations to the mask to remove noise
   kernel = np.ones((5, 5), np.uint8)
   mask = cv2.erode(mask, kernel, iterations=1)
   mask = cv2.dilate(mask, kernel, iterations=1)


   # Find contours in the mask
   contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


   # Draw a bounding box around each detected contour
   for contour in contours:
       print(" detected")
       x, y, w, h = cv2.boundingRect(contour)
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


   # Show the frame with the detected contours
   cv2.imshow('frame', frame)


   # Check if the user has pressed the 'q' key
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
