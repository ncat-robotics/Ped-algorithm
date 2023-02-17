import networkx as nx
import matplotlib.pyplot as plt
class graph:
    def __init__(self,node):
        # Create a graph object
        self.G = nx.Graph()
        self.G_O = nx.Graph()
        self.node = node
 
 ###################################### Add Nodes to Graph #########################################################
    def add_nodes(self):
        for nod in self.node:
            self.G.add_node(nod[0], label=nod[1], distance=nod[2])
            self.G_O.add_node(nod[0], label=nod[1], distance=nod[2])


  ############################## Define Edges to each node ########################
    def create_edges(self):
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
    def shortest_path(self):
            # Find the shortest path between "Red_Ped" and "Green_Ped" nodes
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
                
 ############################### Draw the graph,trees,whatever ##################################
    def draw_raw_graph(self):
        pos = nx.spring_layout(self.G_O)
        nx.draw(self.G_O, pos, with_labels=True)
        labels = nx.get_edge_attributes(self.G_O, 'weight')
        nx.draw_networkx_edge_labels(self.G_O, pos, edge_labels=labels)
        plt.show()
    
    def draw_graph(self):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.show()