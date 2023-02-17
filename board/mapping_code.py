import cv2
import numpy as np


class Map:

    objects = []
    # Define a list of objects
    def __init__(self,ref):
        self.ref = ref
        # Define the robot
        self.robot = {'coordinate': ref, 'label': 'robot', 'color': (128, 128, 128)}

    def insert_nodes(self,nodes):
        for node in nodes:
            self.objects['coordinate'] = node[0]
            self.objects['label'] = node[1]
            if node[1] == 'Red_Pedestal':
                self.objects['color'] = (0,0,255)
            elif node[1] == 'Green_Pedestal':
                self.objects['color'] = (0,255,0)
            elif node[1] == 'Duck':
                self.objects['color'] = (0,255,255)
            elif node[1] == 'White_Pedestal':
                self.objects['color'] = (0,255,255)



    def create_map(self):
        # Read the map image
        img = cv2.imread(r'C:\Users\estde\Documents\Object Detection\Map.png')

        # Draw markers for each object
        for obj in self.objects:
            x, y = obj['coordinate']
            color = obj['color']
            label = obj['label']
            cv2.circle(img, (x, y), 10, color, -1)
            cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw marker for the robot
        x, y = self.robot['coordinate']
        cv2.circle(img, (x, y), 10, (128, 128, 128), -1)
        cv2.putText(img, self.robot['label'], (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return img

    def move_robot(self,img, robot, destination,object, speed=0):
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
        return self.create_map(img, object, robot)

    


    # Create the map with the markers for the objects and robot
    def display_map(self):
        img = self.create_map()

        # Display the map
        cv2.imshow('Map', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_update(self,img):
        # Display the updated map
        cv2.imshow('Map', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
