import numpy as np
import heapq
import math

# Define the dimensions of the arena
width = 8
length = 4

# Define the locations of the statue areas, duck pond, and squares
statue1 = (1,1)
statue2 = (4,2)
statue3 = (7,1)
duck_pond = (3,3)
red_square = (2,2)
green_square = (6,2)

# Define the locations of the pedestals
white_pedestals = [(3,1), (5,1), (6,1)]
green_pedestals = [(4,3), (5,3)]
red_pedestal = (2,1)

# Define the starting position of the robot
robot_pos = (4,0)

# Define a function to calculate the Euclidean distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def astar(start, goal):
    # Define heuristic function as Euclidean distance
    def heuristic(a, b):
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    # Initialize start node and goal node
    start_node = (start[0], start[1])
    goal_node = (goal[0], goal[1])

    # Initialize frontier and explored sets
    frontier = set()
    frontier.add(start_node)
    explored = set()

    # Initialize dictionaries for keeping track of g and f scores, and the previous node in the optimal path
    g_scores = {start_node: 0}
    f_scores = {start_node: heuristic(start_node, goal_node)}
    came_from = {}

    # Keep searching until we've explored every reachable node or found the goal
    while frontier:
        # Choose node in frontier with lowest f score
        current_node = min(frontier, key=lambda node: f_scores[node])

        # If we've reached the goal, reconstruct the path and return it
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append((current_node[0], current_node[1]))
                current_node = came_from[current_node]
            path.append((start_node[0], start_node[1]))
            return list(reversed(path))

        # Move current node from frontier to explored set
        frontier.remove(current_node)
        explored.add(current_node)

        # Check neighbors of current node
        for neighbor in [(current_node[0] + 1, current_node[1]), 
                         (current_node[0] - 1, current_node[1]), 
                         (current_node[0], current_node[1] + 1), 
                         (current_node[0], current_node[1] - 1)]:
            # If neighbor is out of bounds or already explored, skip it
            if neighbor[0] < 0 or neighbor[0] >= 8 or neighbor[1] < 0 or neighbor[1] >= 4:
                continue
            if neighbor in explored:
                continue

            # Calculate tentative g score for neighbor
            tentative_g_score = g_scores[current_node] + 1

            # If neighbor is not in frontier, add it and update g and f scores
            if neighbor not in frontier:
                frontier.add(neighbor)
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                came_from[neighbor] = current_node
            # If neighbor is already in frontier, update its g score and f score if new path is better
            elif tentative_g_score < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                came_from[neighbor] = current_node

    # If we've explored every reachable node and haven't found the goal, return None
    return None

path = astar(robot_pos, white_pedestals[0])
for i in range(len(white_pedestals)-1):
    path += astar(white_pedestals[i], white_pedestals[i+1])
path += astar(white_pedestals[-1], green_pedestals[0])
for i in range(len(green_pedestals)-1):
    path += astar(green_pedestals[i], green_pedestals[i+1])
path += astar(green_pedestals[-1], red_pedestal)
path += astar(red_pedestal, statue1)
# Find the optimal picking and placing order for the ducks
ducks = [(10,1), (3,4), (8,2),(6,7),(5,4), (1,1), (1,2), (3,7), (9,2),(5,5)]
ducks_placed = []
while len(ducks_placed) < len(ducks):
    # Find the optimal path to the nearest remaining duck
    current_pos = duck_pond
    duck_distances = [(i, distance(current_pos, (i%2+2, i//2+1))) for i in ducks if i not in ducks_placed]
    closest_duck = min(duck_distances, key=lambda x: x[1])[0]
    path += astar(current_pos, (closest_duck%2+2, closest_duck//2+1))

    # Pick up the duck and add it to the list of placed ducks
    ducks_placed.append(closest_duck)

# Find the optimal path from the current position to the nearest statue area
current_pos = (closest_duck%2+2, closest_duck//2+1)
path += astar(current_pos, statue1)
statue_distances = [(statue2, distance(current_pos, statue2)),(statue3, distance(current_pos, statue3))]
nearest_statue = min(statue_distances, key=lambda x: x[1])[0]

# Find the optimal path from the current position to the nearest white pedestal
current_pos = nearest_statue
path += astar(current_pos, white_pedestals[0])
pedestal_distances = [(white_pedestals[1:], distance(current_pos, white_pedestals[1])),
                      (green_pedestals, distance(current_pos, green_pedestals[0]))]
nearest_pedestal = min(pedestal_distances, key=lambda x: x[1])[0][0]

# Find the optimal path from the current position to the nearest green pedestal or white pedestal
current_pos = nearest_pedestal
if nearest_pedestal in white_pedestals:
    path += astar(current_pos, green_pedestals[0])
    path += astar(green_pedestals[0], statue2)
    path += astar(statue2, white_pedestals[1])
    path += astar(white_pedestals[1], statue3)
else:
    path += astar(current_pos, statue2)
    path += astar(statue2, white_pedestals[1])
    path += astar(white_pedestals[1], statue3)
    path += astar(statue3, red_square)
    path += astar(red_square, green_square)
print(path)
