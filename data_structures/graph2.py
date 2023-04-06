import networkx as nx

# Create the graph
G = nx.Graph()

# Define the nodes
robot_pos = (4, 1)
white_pedestal_pos = (1, 1)
green_pedestal_pos = (7, 1)
red_pedestal_pos = (4, 3)
statue1_pos = (1, 3)
statue2_pos = (7, 3)
statue3_pos = (4, 2)
duck_pond_pos = (4, 0)
red_square_pos = (6, 0)
green_square_pos = (2, 0)

# Add the nodes to the graph
G.add_nodes_from([
    robot_pos,
    white_pedestal_pos,
    green_pedestal_pos,
    red_pedestal_pos,
    statue1_pos,
    statue2_pos,
    statue3_pos,
    duck_pond_pos,
    red_square_pos,
    green_square_pos
])

# Define the edges and their weights
G.add_edge(robot_pos, white_pedestal_pos, weight=3)
G.add_edge(white_pedestal_pos, green_pedestal_pos, weight=4)
G.add_edge(green_pedestal_pos, red_pedestal_pos, weight=3)
G.add_edge(red_pedestal_pos, statue1_pos, weight=2)
G.add_edge(statue1_pos, duck_pond_pos, weight=1)
G.add_edge(duck_pond_pos, statue2_pos, weight=2)
G.add_edge(statue2_pos, white_pedestal_pos, weight=4)
G.add_edge(white_pedestal_pos, statue3_pos, weight=2)
G.add_edge(statue3_pos, green_pedestal_pos, weight=3)
G.add_edge(green_pedestal_pos, red_square_pos, weight=2)
G.add_edge(red_square_pos, green_square_pos, weight=3)

# Find the optimal path from robot to white pedestal to green pedestal to red pedestal to statue1
path1 = nx.shortest_path(G, source=robot_pos, target=white_pedestal_pos, weight='weight')
path2 = nx.shortest_path(G, source=white_pedestal_pos, target=green_pedestal_pos, weight='weight')
path3 = nx.shortest_path(G, source=green_pedestal_pos, target=red_pedestal_pos, weight='weight')
path4 = nx.shortest_path(G, source=red_pedestal_pos, target=statue1_pos, weight='weight')
optimal_path1 = path1 + path2[1:] + path3[1:] + path4[1:]

# Find the optimal picking and placing order from statue1 to place all ducks in duck pond one at a time
ducks = [(x, y) for x in range(1, 7, 1) for y in range(1, 3, 1)]
optimal_order = []
for duck in ducks:
    path = nx.shortest_path(G, source=statue1_pos, target=duck_pond_pos, weight='weight')
    path2 = nx.shortest_path(G, source=duck_pond_pos, target=statue2_pos, weight='  # ... continued from the previous code ...weight')
    optimal_order.extend(path[1:] + path2[1:])

# From statue3, find the optimal path to stack pedestals in this order ( ->white_pedestal -> green_pedestal -> statue2 -> white_pedestal -> statue3)
path5 = nx.shortest_path(G, source=statue3_pos, target=white_pedestal_pos, weight='weight')
path6 = nx.shortest_path(G, source=white_pedestal_pos, target=green_pedestal_pos, weight='weight')
path7 = nx.shortest_path(G, source=green_pedestal_pos, target=statue2_pos, weight='weight')
path8 = nx.shortest_path(G, source=statue2_pos, target=white_pedestal_pos, weight='weight')
path9 = nx.shortest_path(G, source=white_pedestal_pos, target=statue3_pos, weight='weight')
optimal_path2 = path5 + path6[1:] + path7[1:] + path8[1:] + path9[1:]

# Find the optimal path from statue3 to red square to green square
path10 = nx.shortest_path(G, source=statue3_pos, target=red_square_pos, weight='weight')
path11 = nx.shortest_path(G, source=red_square_pos, target=green_square_pos, weight='weight')
optimal_path3 = path10 + path11[1:]

# Print the optimal paths and orders
print(f"Optimal path from robot to ducks: {optimal_path1}")
print(f"Optimal duck picking and placing order: {optimal_order}")
print(f"Optimal path to stack pedestals: {optimal_path2}")
print(f"Optimal path to visit red and green squares: {optimal_path3}")
