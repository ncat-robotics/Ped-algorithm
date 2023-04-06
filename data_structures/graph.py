import networkx as nx
import matplotlib.pyplot as plt
import math


class Graph:
  def __init__(self, nodes):
        # Create a graph object
        self.G = nx.Graph()
        self.nodes = nodes
        self.edges = []
        self.white_peds = []
        self.red_peds = []
        self.green_peds = []
        self.ducks = []
        
        
        # Add Nodes to Graph
        for node in self.nodes:
            if node[1] == "White_Pedestal":
                self.white_peds.append(node[2])
            elif node[1] == "Red_Pedestal":
                self.red_peds.append(node[2])
            elif node[1] == "Green_Pedestal":
                self.green_peds.append(node[2])
            elif node[1] == "Duck":
                self.ducks.append(node[2])


        self.white_peds.sort()
        self.red_peds.sort()
        self.green_peds.sort()
       
        print(self.ducks)
        self.robot_pos = (4, 1)
        self.white_pedestal_pos1 = self.white_peds[0]
        self.white_pedestal_pos2 = self.white_peds[1]
        self.white_pedestal_pos3 = self.white_peds[2]
        self.green_pedestal_pos1 = self.green_peds[0]
        self.green_pedestal_pos2 = self.green_peds[1]
        self.red_pedestal_pos = self.red_peds[0]
        self.statue1_pos = (1, 3)
        self.statue2_pos = (7, 3)
        self.statue3_pos = (4, 2)
        self.duck_pond_pos = (4, 0)
        self.red_square_pos = (6, 0)
        self.green_square_pos = (2, 0)

        
        
        self.G.add_nodes_from([
            self.robot_pos,
            self.white_pedestal_pos1,
            self.white_pedestal_pos2,
            self.white_pedestal_pos3,
            self.green_pedestal_pos1,
            self.green_pedestal_pos2,
            self.red_pedestal_pos,
            self.statue1_pos,
            self.statue2_pos,
            self.statue3_pos,
            self.duck_pond_pos,
            self.red_square_pos,
            self.green_square_pos
            
        ])

        self.G.add_nodes_from(
            self.ducks
        )
        
           
  


   ############################################## Find Shortest Path ###################################G
  def shortest_path(self):
       # Define the edges and their weights
        self.G.add_edge(self.robot_pos, self.white_pedestal_pos1, weight= math.sqrt((pow((self.white_pedestal_pos1[0]-self.robot_pos[0]),2) + pow((self.white_pedestal_pos1[1]-self.robot_pos[1]),2))))
        self.G.add_edge(self.robot_pos, self.white_pedestal_pos2, weight= math.sqrt((pow((self.white_pedestal_pos2[0]-self.robot_pos[0]),2) + pow((self.white_pedestal_pos2[1]-self.robot_pos[1]),2))))
        self.G.add_edge(self.robot_pos, self.white_pedestal_pos3, weight= math.sqrt((pow((self.white_pedestal_pos3[0]-self.robot_pos[0]),2) + pow((self.white_pedestal_pos3[1]-self.robot_pos[1]),2))))

        self.G.add_edge(self.white_pedestal_pos1, self.green_pedestal_pos1, weight= math.sqrt((pow((self.green_pedestal_pos1[0]-self.white_pedestal_pos1[0]),2) + pow((self.green_pedestal_pos1[1]-self.white_pedestal_pos1[1]),2))))
        self.G.add_edge(self.white_pedestal_pos1, self.green_pedestal_pos2, weight= math.sqrt((pow((self.green_pedestal_pos2[0]-self.white_pedestal_pos2[0]),2) + pow((self.green_pedestal_pos2[1]-self.white_pedestal_pos2[1]),2))))
        self.G.add_edge(self.white_pedestal_pos1, self.statue3_pos, weight= math.sqrt((pow((self.statue3_pos[0]-self.white_pedestal_pos1[0]),2) + pow((self.statue3_pos[1]-self.white_pedestal_pos1[1]),2))))

        self.G.add_edge(self.white_pedestal_pos2, self.green_pedestal_pos2, weight= math.sqrt((pow((self.green_pedestal_pos2[0]-self.white_pedestal_pos2[0]),2) + pow((self.green_pedestal_pos2[1]-self.white_pedestal_pos2[1]),2))))
        self.G.add_edge(self.white_pedestal_pos2, self.green_pedestal_pos1, weight= math.sqrt((pow((self.green_pedestal_pos1[0]-self.white_pedestal_pos2[0]),2) + pow((self.green_pedestal_pos1[1]-self.white_pedestal_pos2[1]),2))))
        self.G.add_edge(self.white_pedestal_pos2, self.statue3_pos, weight= math.sqrt((pow((self.statue3_pos[0]-self.white_pedestal_pos2[0]),2) + pow((self.statue3_pos[1]-self.white_pedestal_pos2[1]),2))))

        self.G.add_edge(self.white_pedestal_pos3, self.green_pedestal_pos2, weight= math.sqrt((pow((self.green_pedestal_pos2[0]-self.white_pedestal_pos3[0]),2) + pow((self.green_pedestal_pos2[1]-self.white_pedestal_pos3[1]),2)))  )
        self.G.add_edge(self.white_pedestal_pos3, self.green_pedestal_pos1, weight= math.sqrt((pow((self.green_pedestal_pos1[0]-self.white_pedestal_pos3[0]),2) + pow((self.green_pedestal_pos1[1]-self.white_pedestal_pos3[1]),2))))
        self.G.add_edge(self.white_pedestal_pos3, self.statue3_pos, weight= math.sqrt((pow((self.statue3_pos[0]-self.white_pedestal_pos3[0]),2) + pow((self.statue3_pos[1]-self.white_pedestal_pos3[1]),2))))

        self.G.add_edge(self.green_pedestal_pos1, self.red_pedestal_pos, weight= math.sqrt((pow((self.red_pedestal_pos[0]-self.green_pedestal_pos1[0]),2) + pow((self.red_pedestal_pos[1]-self.green_pedestal_pos1[1]),2))))
        self.G.add_edge(self.green_pedestal_pos1, self.statue2_pos, weight= math.sqrt((pow((self.statue2_pos[0]-self.green_pedestal_pos1[0]),2) + pow((self.statue2_pos[1]-self.green_pedestal_pos1[1]),2))))

        self.G.add_edge(self.green_pedestal_pos2, self.red_pedestal_pos, weight= math.sqrt((pow((self.red_pedestal_pos[0]-self.green_pedestal_pos2[0]),2) + pow((self.red_pedestal_pos[1]-self.green_pedestal_pos2[1]),2))) )
        self.G.add_edge(self.green_pedestal_pos2, self.statue2_pos, weight= math.sqrt((pow((self.statue2_pos[0]-self.green_pedestal_pos2[0]),2) + pow((self.statue2_pos[1]-self.green_pedestal_pos2[1]),2))))

        self.G.add_edge(self.red_pedestal_pos, self.statue1_pos, weight= math.sqrt((pow((self.statue1_pos[0]-self.red_pedestal_pos[0]),2) + pow((self.statue2_pos[1]-self.red_pedestal_pos[1]),2))))

        self.G.add_edge(self.statue1_pos, self.duck_pond_pos, weight= math.sqrt((pow((self.duck_pond_pos[0]-self.statue1_pos[0]),2) + pow((self.duck_pond_pos[1]-self.statue1_pos[1]),2))))

        self.G.add_edge(self.duck_pond_pos, self.statue2_pos, weight= math.sqrt((pow((self.duck_pond_pos[0]-self.statue2_pos[0]),2) + pow((self.duck_pond_pos[1]-self.statue2_pos[1]),2))))

        self.G.add_edge(self.statue2_pos, self.white_pedestal_pos2, weight= math.sqrt((pow((self.white_pedestal_pos2[0]-self.statue2_pos[0]),2) + pow((self.white_pedestal_pos2[1]-self.statue2_pos[1]),2))))

        self.G.add_edge(self.statue3_pos, self.red_square_pos, weight= math.sqrt((pow((self.red_square_pos[0]-self.statue3_pos[0]),2) + pow((self.red_square_pos[1]-self.statue3_pos[1]),2))))

        self.G.add_edge(self.red_square_pos, self.green_square_pos, weight= math.sqrt((pow((self.green_square_pos[0]-self.red_square_pos[0]),2) + pow((self.green_square_pos[1]-self.red_square_pos[1]),2))))

        for duck in self.ducks:
             self.G.add_edge(self.statue1_pos, duck, weight= math.sqrt((pow((duck[0]-self.statue1_pos[0]),2) + pow((duck[1]-self.statue1_pos[1]),2))))
             self.G.add_edge(self.duck_pond_pos, duck, weight= math.sqrt((pow((duck[0]-self.duck_pond_pos[0]),2) + pow((duck[1]-self.duck_pond_pos[1]),2))))

        
       
        # Find the optimal path from robot to white pedestal to green pedestal to red pedestal to statue1
        path1 = nx.shortest_path(self.G, source=self.robot_pos, target=self.white_pedestal_pos1, weight='weight')
        path2 = nx.shortest_path(self.G, source=self.white_pedestal_pos1, target=self.green_pedestal_pos1, weight='weight')
        path3 = nx.shortest_path(self.G, source=self.green_pedestal_pos1, target=self.red_pedestal_pos, weight='weight')
        path4 = nx.shortest_path(self.G, source=self.red_pedestal_pos, target=self.statue1_pos, weight='weight')
        optimal_path1 = path1 + path2[1:] + path3[1:] + path4[1:]

        # Find the optimal picking and placing order from statue1 to place all ducks in duck pond one at a time
        optimal_order = []
        for duck in self.ducks:
            path = nx.shortest_path(self.G, source=self.statue1_pos, target=self.duck_pond_pos, weight='weight')
            path2 = nx.shortest_path(self.G, source=self.duck_pond_pos, target=duck, weight='weight')
            optimal_order.extend(path[1:] + path2[1:])

        # From statue3, find the optimal path to stack pedestals in this order ( ->white_pedestal -> green_pedestal -> statue2 -> white_pedestal -> statue3)
        path5 = nx.shortest_path(self.G, source=self.statue3_pos, target=self.white_pedestal_pos2, weight='weight')
        path6 = nx.shortest_path(self.G, source=self.white_pedestal_pos2, target=self.green_pedestal_pos2, weight='weight')
        path7 = nx.shortest_path(self.G, source=self.green_pedestal_pos2, target=self.statue2_pos, weight='weight')
        path8 = nx.shortest_path(self.G, source=self.statue2_pos, target=self.white_pedestal_pos2, weight='weight')
        path9 = nx.shortest_path(self.G, source=self.white_pedestal_pos2, target=self.statue3_pos, weight='weight')
        optimal_path2 = path5 + path6[1:] + path7[1:] + path8[1:] + path9[1:]

        # Find the optimal path from statue3 to red square to green square
        path10 = nx.shortest_path(self.G, source=self.statue3_pos, target=self.red_square_pos, weight='weight')
        path11 = nx.shortest_path(self.G, source=self.red_square_pos, target=self.green_square_pos, weight='weight')
        optimal_path3 = path10 + path11[1:]

        # Print the optimal paths and orders
        path1 =[]
        path1_translations =[]
        path2_translations =[]
        path3_translations =[]
        path4_translations =[]
        path2 =[]
        path3 =[]
        path4 =[]
        i = 0
        while i < len(optimal_path1) -1 :
            path1.append(self.G.get_edge_data(optimal_path1[i],optimal_path1[i+1]))
            path1_translations.append(((optimal_path1[i+1][0] - (optimal_path1[i])[0]),(optimal_path1[i+1][1] - (optimal_path1[i])[1])))
            i = i +1
        print(f"Optimal path from robot to ducks: {optimal_path1}")
        print("Diagonals")
        print(path1)
        print("Translations")
        print(path1_translations)
        i = 0
        while i < len(optimal_order) -1 :
            path2.append(self.G.get_edge_data(optimal_order[i],optimal_order[i+1]))
            path2_translations.append(((optimal_order[i+1][0] - (optimal_order[i])[0]),(optimal_order[i+1][1] - (optimal_order[i])[1])))
            i = i +1
        print(f"Optimal duck picking and placing order: {optimal_order}")
        print("Diagonals")
        print(path2)
        print("Translations")
        print(path2_translations)
        i = 0
        while i < len(optimal_path2) -1 :
            path3.append(self.G.get_edge_data(optimal_path2[i],optimal_path2[i + 1]))
            path3_translations.append(((optimal_path2[i+1][0] - (optimal_path2[i])[0]),(optimal_path2[i+1][1] - (optimal_path2[i])[1])))
            i = i + 1

        print(f"Optimal path to stack pedestals: {optimal_path2}")
        print("Diagonals")
        print(path3)
        print("Translations")
        print(path3_translations)
        i = 0
        while i < len(optimal_path3) -1 :
            path4.append(self.G.get_edge_data(optimal_path3[i],optimal_path3[ i + 1]))
            path4_translations.append(((optimal_path3[i+1][0] - (optimal_path3[i])[0]),(optimal_path3[i+1][1] - (optimal_path3[i])[1])))
            i = i + 1
        print(f"Optimal path to visit red and green squares: {optimal_path3}")
        print("Diagonals")
        print(path4)
        print("Translations")
        print(path4_translations)

        translations = path1_translations + path2_translations[1:] + path3_translations[1:] + path4_translations

        return translations
       
        



  def draw_graph(self):
      pos = nx.spring_layout(self.G)
      #nx.bfs
      nx.draw(self.G, pos, with_labels=True)
      labels = nx.get_edge_attributes(self.G, 'weight')
      nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
      plt.show()


