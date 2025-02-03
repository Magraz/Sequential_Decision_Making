from provided_code.maze import *
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import time
import os

dirname = os.path.dirname(__file__)

maze1_filename = os.path.join(dirname, "provided_code/maze1.pgm")

maze2_filename = os.path.join(dirname, "provided_code/maze2.pgm")

maze_1 = Maze2D.from_pgm(maze1_filename)

# maze_1.start_state = (1, 1)
# maze_1.start_index = maze_1.index_from_state(maze_1.start_state)

# maze_1.goal_state = (maze_1.cols, maze_1.rows)
# maze_1.goal_index = maze_1.index_from_state(maze_1.goal_state)

maze_2 = Maze2D.from_pgm(maze2_filename)

# maze_2.start_state = (1, 1)
# maze_2.start_index = maze_2.index_from_state(maze_2.start_state)

# maze_2.goal_state = (maze_2.cols, maze_2.rows)
# maze_2.goal_index = maze_2.index_from_state(maze_2.goal_state)


def steer(x_nearest, x_rand, step_size):  # steer function
    direction = np.array(x_rand) - np.array(x_nearest)  # get the direction
    distance = np.linalg.norm(direction)  # get the distance
    if distance == 0:  # if the distance is 0
        return x_nearest  # return the nearest point
    direction = direction / distance  # normalize the direction
    x_new = np.array(x_nearest) + step_size * direction  # get the new point
    return tuple(x_new)  # return the new point


def is_obstacle_free(
    maze, start, end, num_points=20
):  # check if the path is obstacle free
    for i in range(num_points + 1):  # for each point
        x = start[0] + (end[0] - start[0]) * i / num_points  # get the x coordinate
        y = start[1] + (end[1] - start[1]) * i / num_points  # get the y coordinate
        # Round to nearest grid cell
        grid_x, grid_y = round(x), round(y)  # round the coordinates
        if maze.check_occupancy((grid_x, grid_y)):  # if the cell is occupied
            return False  # return False
    return True  # return True


def calculate_path_length(path):  # calculate the path length
    length = 0  # initilize the length
    for i in range(len(path) - 1):  # for each point
        length += np.linalg.norm(
            np.array(path[i + 1]) - np.array(path[i])
        )  # calculate the distance
    return length  # return the length


def rrt(maze, step_size=0.1, goal_sample_rate=0.1, max_iters=1000):
    start_time = time.time()  # Start time
    start = maze.state_from_index(maze.get_start())  # Get the start state
    goal = maze.state_from_index(maze.get_goal())  # Get the goal state

    vertices = [start]  # Initialize the vertices
    edges = {}  # Initialize the edges
    nn = NearestNeighbors(
        n_neighbors=1, algorithm="auto"
    )  # Initialize nearest neighbors

    for _ in range(max_iters):  # For each iteration
        nn.fit(vertices)  # Fit the nearest neighbors

        # Sample a random point or the goal with some probability
        if random.random() < goal_sample_rate:
            x_rand = goal  # Set the random point to the goal
        else:
            x_rand = (
                random.uniform(0, maze.cols - 1),
                random.uniform(0, maze.rows - 1),
            )  # Random floating-point point

        # Find the nearest neighbor
        _, indices = nn.kneighbors([x_rand])  # Get the nearest neighbor
        x_nearest = vertices[indices[0][0]]  # Retrieve the nearest vertex
        x_new = steer(x_nearest, x_rand, step_size)  # Compute the new point

        # Skip if the path is not obstacle-free
        if not is_obstacle_free(maze, x_nearest, x_new):
            continue

        # Add the new vertex and record the edge
        vertices.append(x_new)
        edges[tuple(round(coord, 5) for coord in x_new)] = tuple(
            round(coord, 5) for coord in x_nearest
        )  # Add the new vertex to the tree wiht the parent

        # Check if the goal is reached
        if (
            np.linalg.norm(np.array(x_new) - np.array(goal)) <= 1.0
        ):  # if the goal is reached
            print("Goal reached!")

            # Add the goal explicitly and record its parent
            vertices.append(goal)
            edges[tuple(round(coord, 5) for coord in goal)] = tuple(
                round(coord, 5) for coord in x_new
            )  # Add the goal to the tree with the parent

            # Path reconstruction
            path = [goal]
            while tuple(round(coord, 5) for coord in path[-1]) != tuple(
                round(coord, 5) for coord in start
            ):  # while the path is not the start point with rounding
                found_parent = False
                for child, parent in edges.items():  # for each edge
                    if tuple(round(coord, 5) for coord in child) == tuple(
                        round(coord, 5) for coord in path[-1]
                    ):  # if the child is the last point
                        path.append(parent)  # add the parent
                        found_parent = True  # set found parent to True
                        break
                if not found_parent:  # if no parent is found
                    print("No parent found for path reconstruction")
                    return None, None, None
            path.reverse()  # Reverse the path to start-to-goal order

            runtime = time.time() - start_time  # Compute runtime
            path_length = calculate_path_length(path)  # Compute path length
            return path, path_length, runtime

    print("No path found")  # If max_iters is reached, print failure
    runtime = time.time() - start_time  # Compute runtime
    return None, None, runtime  # Return failure and runtime


path, path_length, runtime = rrt(
    maze=maze_1, step_size=1.0, goal_sample_rate=0.5, max_iters=10000
)

print(path_length)

maze_1.plot_path(path)
