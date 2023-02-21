import cv2
import numpy as np


class Map:

    
   def __init__(self, ref):
        self.ref = ref
        self.objects = []
        self.robot = {'coordinate': ref, 'label': 'robot', 'color': (128, 128, 128)}

   def insert_nodes(self, nodes):
        for node in nodes:
            obj = {'coordinate': node[0], 'label': node[1]}
            if node[1] == 'Red_Pedestal':
                obj['color'] = (0,0,255)
            elif node[1] == 'Green_Pedestal':
                obj['color'] = (0,255,0)
            elif node[1] == 'Duck':
                obj['color'] = (0,255,255)
            elif node[1] == 'White_Pedestal':
                obj['color'] = (255,255,255)
            self.objects.append(obj)



   def create_map(self):
        img = cv2.imread('/home/autodrone-hardware/Documents/GitHub/Ped-algorithm/board/Map.png')
        for obj in self.objects:
            y = int(obj['coordinate'][1])
            x = int(obj['coordinate'][0])
            color = obj['color']
            label = obj['label']
            cv2.circle(img, (x, y), 10, color, -1)
            cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        x, y = self.robot['coordinate']
        cv2.circle(img, (x, y), 10, (128, 128, 128), -1)
        cv2.putText(img, self.robot['label'], (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return img

   def move_robot(self,img, robot, destination,object, speed=0):
        # Calculate the distance between the current position and destination
        x1, y1 = self.robot['coordinate']
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
