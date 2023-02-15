import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist
import config
import math
import tree
from tree import *
import networkx as nx
import matplotlib.pyplot as plt

 
CLASSES = config.CLASSES
 
class help:
 
 def __init__(self,ref):
   self.ref = ref
   # Create a graph object
   self.G = nx.Graph()
   self.G_O = nx.Graph()
 
 def new_reference(self,ref):
   self.ref = ref
 travel_x = 0
 travel_y = 0
 D = 0
 node = []
 sorted_data = []
 group = []
 groups = []
 tree_dist = None
 edges = []
 # Define a list of colors for visualization
 COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)
 
 def midpoint(self,ptA, ptB):
   return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 
 def preprocess_image_2(self,image_path, input_size):
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
 
 
 def detect_objects_2(self,interpreter, image, threshold):
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
 
 
 def run_odt_and_draw_results_2(self,image_path, interpreter, threshold=0.5):
   ''' Run object detection on the input image and draw the detection results
   '''
 
   moves = []
  
   # Load the input shape required by the model
   _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
 
   # Load the input image and preprocess it
   preprocessed_image, original_image = self.preprocess_image_2(
       image_path,
       (input_height, input_width)
     )
 
   # Run object detection on the input image
   results = self.detect_objects_2(interpreter, preprocessed_image, threshold=threshold)
 
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
     x_ratio = 0.021 * math.cos(20)
     y_ratio = 0.0421 #* math.cos(20)
     start_x = self.ref[0]
     start_y = self.ref[1]
 
     real_start_x = start_x * x_ratio
     real_start_y = start_y * y_ratio
     # Draw the bounding box and label on the image
     color = [int(c) for c in self.COLORS[class_id]]
     cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
     # Make adjustments to make the label visible for all objects
     y = ymin - 15 if ymin - 15 > 15 else ymin + 15
     label = "{}: {:.0f}%".format(CLASSES[class_id], obj['score'] * 100)
     cv2.putText(original_image_np, label, (xmin, y),
         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
     # find midpoint of bounding box
     (mX, mY) = self.midpoint((xmin, ymin), (xmax, ymax))
 
     # Get the real dimensions in cm
     real_mX = mX * x_ratio
     real_mY = mY * y_ratio
 
     #Direction + or - from the ref point(origin)
     if(mX < start_x):
       self.travel_x= -real_mX
     if(mX >= start_x):
       self.travel_x = real_mX
 
     if(mY< start_y):
       self.travel_y = -real_mY
     if(mX >= start_x):
       self.travel_y = real_mY
 
     # Prints where the robot needs to travel
     print("Travel ")
     print([self.travel_x,self.travel_y])
 
     # draws the distance of the ref point to the midpoint of the bounding box
     cv2.circle(original_image_np, (start_x,start_y), 5, color, -1)
     cv2.circle(original_image_np, (int(mX), int(mY)), 5, color, -1)
     cv2.line(original_image_np, (start_x,start_y), (int(mX), int(mY)),
       color, 2)
    
     #Calculates the diangonal distance of ref point to midpoint
     self.D =  math.sqrt((pow((real_start_x-real_mX),2) + pow((real_start_y- real_mY),2)))
     print("Distance: ")
     print(self.D)
 
     # Adds the detetections label, distance, and coordinate to a linked list
     if(class_id == 0 or class_id == 7 or class_id == 8 or class_id == 9 or class_id == 10 ):
       self.node.append([self.D,"Duck",[self.travel_x,self.travel_y]])
     elif(class_id == 3 or class_id == 4 ):
       self.node.append([self.D,"Green_Pedestal",[self.travel_x,self.travel_y]])
     elif(class_id == 5 or class_id == 6 ):
       self.node.append([self.D,"Red_Pedestal",[self.travel_x,self.travel_y]])

 
    
     # Labels distance on the the map image
     cv2.putText(original_image_np, "{:.1f}cm".format(self.D), (int(mX), int(mY - 10)),
       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

  ###################################### Add Nodes to Graph #########################################################
   for nod in self.node:
    self.G.add_node(nod[0], label=nod[1], distance=nod[2])
    self.G_O.add_node(nod[0], label=nod[1], distance=nod[2])


  ############################## Define Edges to each node ########################
   i = 1
   for X in self.node:
    j = i
    while j < len(self.node):
      temp_distance = X[0] - self.node[j][0]
      self.edges.append([X[0],self.node[j][0],temp_distance])
      j = j +1
    i = i +1
   # Add the edges to the graph
   for edge in self.edges:
      self.G.add_edge(edge[0], edge[1], weight=edge[2])
      self.G_O.add_edge(edge[0], edge[1], weight=edge[2])

  ##############################################Find Shortest Path ###################################
  # Find the shortest path between "Red_Ped" and "Green_Ped" nodes
   try:
    source = [nod[0] for nod in self.node if nod[1] == "Red_Pedestal"][0]
    print("source: ")
    print(source)
    target = [nod[0] for nod in self.node if nod[1] == "Green_Pedestal"][0]
    shortest_path = nx.dijkstra_path(self.G, source, target, weight='weight')
    print("Shortest path between Red_Ped and Green_Ped:", shortest_path)
    # Find the shortest path between "Green_Ped" and "White_Ped" nodes
    for node in self.node:
      if node[0] == shortest_path[1] and node[1] =="Green_Ped":
          source = node[0]
   finally:


    ############################# Sorting Section #########################
    # creates and prints the final binary search tree
    for X in self.node:
      self.tree_dist = insert(self.tree_dist,X[0],X[1],X[2])
    print("Tree ")
    inorder(self.tree_dist)
    inorder_grouping(self.tree_dist,self.group,self.groups)
    
    #OR
    def sort_priority(elem):
      return elem[0]

    for obj in self.node:
      self.sorted_data = sorted(self.node,key = sort_priority, reverse = False)

      #OR

    # Return the final map image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8
 ############################### Draw the graph,trees,whatever ##################################
 def draw_raw_graph(self):
   pos = nx.spring_layout(self.G_O)
   nx.draw(self.G_O, pos, with_labels=True)
   labels = nx.get_edge_attributes(self.G_O, 'weight')
   nx.draw_networkx_edge_labels(self.G_O, pos, edge_labels=labels)
   plt.show()
   
 def draw_raw_graph(self):
   pos = nx.spring_layout(self.G)
   nx.draw(self.G, pos, with_labels=True)
   labels = nx.get_edge_attributes(self.G, 'weight')
   nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
   plt.show()

 def return_tree(self):
   values = []
   values.append(inorder_return(self.tree_dist))
   return values

 def return_sorted(self):
   values = []
   values.append(self.sorted_data)
   print("Sorted: ")
   print(self.sorted_data)

   return values
 
 
 
 

"""
   target = [node[0] for node in self.node if node[1] == "White_Ped"][0]
   shortest_path_2 = nx.dijkstra_path(G, source, target, weight='weight')
   print("Shortest path between Green_Ped and White_Ped:", shortest_path_2)
   for x in shortest_path_2:    
     self.G.remove_node(x)
     for node in self.node:
        if x == node[0]:
            self.node.remove(node)
           
   travel = []
   travel.append(shortest_path_2)
   print("Travel: ")
   print(travel)




   # Find the shortest path between "Green_Ped" and "White_Ped" nodes
   source = [node[0] for node in self.node if node[1] == "Green_Ped"][0]
   print("source: ")
   print(source)
   target = [node[0] for node in self.node if node[1] == "White_Ped"][0]
   shortest_path = nx.dijkstra_path(self.G, source, target, weight='weight')
   print("Shortest path between Green_Ped and White_Ped:", shortest_path)


   travel.append(shortest_path)
   print("Travel: ")
   print(travel)


   for x in shortest_path:    
    self.G.remove_node(x)
    for node in self.node:
        if x == node[0]:
            self.node.remove(node)
"""