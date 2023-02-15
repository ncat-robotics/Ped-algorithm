import networkx as nx
import matplotlib.pyplot as plt

# Create a graph object
G = nx.Graph()
G_O = nx.Graph()

# Define the nodes and their labels
# Assume the nodes and their distances have been pre-defined
nodes = [(1, "Red_Ped", 0), (2, "White_Ped", 5), (3, "White_Ped", 10), (4, "Green_Ped", 3), (5, "Duck", 15), (6, "Duck", 20), (7, "White_Ped", 4), (8, "Duck", 25), (9, "Duck", 30), (10, "Duck", 35), (11, "Green_Ped", 1), (12, "Duck", 40)]
buffer = [(1, "Red_Ped", 0), (2, "White_Ped", 5), (3, "White_Ped", 10), (4, "Green_Ped", 3), (5, "Duck", 15), (6, "Duck", 20), (7, "White_Ped", 4), (8, "Duck", 25), (9, "Duck", 30), (10, "Duck", 35), (11, "Green_Ped", 1), (12, "Duck", 40)]
# Add the nodes to the graph
for node in nodes:
    G.add_node(node[0], label=node[1], distance=node[2])
    G_O.add_node(node[0], label=node[1], distance=node[2])

# Define the edges and their weights
# Assume the distances between nodes have been pre-defined
edges = [(1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 7, 4), (3, 7, 6), (4, 5, 15), (4, 11, 1), (5, 6, 5), (6, 7, 5), (7, 8, 10), (8, 9, 5), (9, 10, 5), (10, 11, 20), (11, 12, 25)]

# Add the edges to the graph
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])
    G_O.add_edge(edge[0], edge[1], weight=edge[2])

# Find the shortest path between "Red_Ped" and "Green_Ped" nodes
source = [node[0] for node in nodes if node[1] == "Red_Ped"][0]
print("source: ")
print(source)
target = [node[0] for node in nodes if node[1] == "Green_Ped"][0]
shortest_path = nx.dijkstra_path(G, source, target, weight='weight')
print("Shortest path between Red_Ped and Green_Ped:", shortest_path)

# Find the shortest path between "Green_Ped" and "White_Ped" nodes
for node in nodes:
    if node[0] == shortest_path[1] and node[1] =="Green_Ped":
        source = node[0]

target = [node[0] for node in nodes if node[1] == "White_Ped"][0]
shortest_path_2 = nx.dijkstra_path(G, source, target, weight='weight')
print("Shortest path between Green_Ped and White_Ped:", shortest_path_2)
for x in shortest_path_2:    
    G.remove_node(x)
    for node in nodes:
        if x == node[0]:
            nodes.remove(node)
            
travel = []
travel.append(shortest_path_2)
print("Travel: ")
print(travel)
# Find the strongly connected components
#strongly_connected = nx.strongly_connected_components(G)
#print("Strongly connected components:", list(strongly_connected))

# Find the shortest path between "Green_Ped" and "White_Ped" nodes
source = [node[0] for node in nodes if node[1] == "Green_Ped"][0]
print("source: ")
print(source)
target = [node[0] for node in nodes if node[1] == "White_Ped"][0]
shortest_path = nx.dijkstra_path(G, source, target, weight='weight')
print("Shortest path between Green_Ped and White_Ped:", shortest_path)

travel.append(shortest_path)
print("Travel: ")
print(travel)

for x in shortest_path:    
    G.remove_node(x)
    for node in nodes:
        if x == node[0]:
            nodes.remove(node)

"""
for x in buffer: 
    if x[1] != "Duck":   
        G_O.remove_node(x[0])
        buffer.remove(x)
"""



# Draw the graph
pos = nx.spring_layout(G_O)
nx.draw(G_O, pos, with_labels=True)
labels = nx.get_edge_attributes(G_O, 'weight')
nx.draw_networkx_edge_labels(G_O, pos, edge_labels=labels)
plt.show()
