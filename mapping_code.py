import cv2
import numpy as np

# Define a list of objects
objects = [
    {'coordinate': (100, 50), 'label': 'duck', 'color': (0, 255, 255)},
    {'coordinate': (150, 80), 'label': 'duck', 'color': (0, 255, 255)},
    {'coordinate': (200, 30), 'label': 'duck', 'color': (0, 255, 255)},
    {'coordinate': (30, 30), 'label': 'red_ped', 'color': (0, 0, 255)}

]

# Define the robot
robot = {'coordinate': (250, 240), 'label': 'robot', 'color': (128, 128, 128)}

def create_map(map_image, objects, robot):
    # Read the map image
    img = cv2.imread(r'C:\Users\estde\Documents\Object Detection\Map.png')

    # Draw markers for each object
    for obj in objects:
        x, y = obj['coordinate']
        color = obj['color']
        label = obj['label']
        cv2.circle(img, (x, y), 10, color, -1)
        cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw marker for the robot
    x, y = robot['coordinate']
    cv2.circle(img, (x, y), 10, (128, 128, 128), -1)
    cv2.putText(img, robot['label'], (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def move_robot(img, robot, destination,object, speed=0):
    # Calculate the distance between the current position and destination
    x1, y1 = robot['coordinate']
    x2, y2 = destination
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # If the speed is set, calculate the time it takes to move
    if speed != 0:
        time = distance / speed
    else:
        time = 0

    # Update the robot's position
    robot['coordinate'] = destination

    # Check if the new position of the robot matches any of the objects
    object = [obj for obj in object if obj['coordinate'] != destination]

    # Redraw the map with the updated robot position and remove the markers for the collected objects
    return create_map(img, object, robot)

# Define a map image
map_image = r'C:\Users\estde\Documents\Object Detection\Map.png'


# Create the map with the markers for the objects and robot
img = create_map(map_image, objects, robot)

# Display the map
cv2.imshow('Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Move the robot to a new position
destination = (150, 50)
img = move_robot(img, robot, destination,objects, speed=10)

# Display the updated map
cv2.imshow('Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
