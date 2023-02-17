import cv2
import numpy as np

# Read the input image
img = cv2.imread(r'C:\Users\estde\Documents\Object Detection\document_scanner\Ped-algorithm\test_images\20221101_182957.jpg')

img = cv2.resize(img, (800, 800))

# define the alpha and beta
alpha = 1.5# Contrast control
beta = 20 # Brightness control

# call convertScaleAbs function
img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Convert the image from BGR to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper limits of the yellow color in HSV color space
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])


# Define range for green color
green_lower = np.array([50,100,100])
green_upper = np.array([70,255,255])

# Define range for red color
red_lower = np.array([0,100,100])
red_upper = np.array([10,255,255])

# Threshold the image to get only the yellow color
yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

# Threshold the image to get only the green color
green_mask = cv2.inRange(hsv_img, green_lower, green_upper)

# Threshold the image to get only the green color
red_mask = cv2.inRange(hsv_img, red_lower, red_upper)

# Find the yellow contours in the binary image
contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding rectangle around the yellow object
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 225, 255), 2)

    # Get the color hue of the yellow object
    hue = hsv_img[y + h // 2][x + w // 2][0]

    # Add a label of "Duck" and the color hue value to the image
    if (w*h) > 1600:
        cv2.putText(img, "Duck", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Hue: " + str(hue), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Size: " + str(w*h), (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# Find the green contours in the binary image
contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding rectangle around the yellow object
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Get the color hue of the yellow object
    hue = hsv_img[y + h // 2][x + w // 2][0]

    # Add a label of "Duck" and the color hue value to the image
    if (w*h) > 1600:
        cv2.putText(img, "Green_Ped", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Hue: " + str(hue), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Size: " + str(w*h), (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Find the contours in the binary image
contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding rectangle around the yellow object
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Get the color hue of the yellow object
    hue = hsv_img[y + h // 2][x + w // 2][0]

    # Add a label of "Duck" and the color hue value to the image
    if (w*h) > 1900:
        cv2.putText(img, "Red_Ped", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Hue: " + str(hue), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Size: " + str(w*h), (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



# Show the output image
cv2.imshow("Detected Yellow Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()