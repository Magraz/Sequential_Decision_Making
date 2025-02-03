from provided_code.maze import Maze2D, Maze4D, Maze
from provided_code.priority_queue import PriorityQueue
import os
import numpy as np
from copy import deepcopy
import time
import random
from sklearn.neighbors import NearestNeighbors

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


class Node:
    def __init__(
        self, parent, index: int, state: tuple, leaf: bool = False, root: bool = False
    ):
        self.parent = parent
        self.index = index
        self.state = state


def RRT(
    maze: Maze2D | Maze4D,
    step_size: int = 1,
    max_steps: int = 10000,
    goal_sample_rate: float = 0.2,
):
    T = {}
    q_start = Node(parent=None, index=maze.start_index, state=maze.start_state)
    T[q_start.index] = q_start

    start_time = time.time()

    path = []
    for i in range(max_steps):

        # Sample a random point or the goal with some probability
        if random.random() < goal_sample_rate:
            q_rand_idx = maze.goal_index  # Set the random point to the goal
        else:
            q_rand_idx = np.random.randint(
                low=0, high=maze.index_from_state((maze.cols - 1, maze.rows - 1))
            )

        q_rand_state = np.expand_dims(
            np.array(maze.state_from_index(q_rand_idx)), axis=0
        )

        tree_states = np.array([node.state for node in T.values()])
        nns = NearestNeighbors(n_neighbors=1).fit(tree_states)

        _, indices = nns.kneighbors(q_rand_state)

        q_near_state = np.squeeze(tree_states[indices], axis=0)

        direction = q_rand_state - q_near_state
        dist = np.linalg.norm(direction)
        if dist == 0:
            continue
        else:
            unit_direction = direction / dist
            q_new_state = q_near_state + np.round(unit_direction * step_size)

        q_new_state = tuple(q_new_state[0, :].tolist())

        q_near_state = tuple(q_near_state[0, :].tolist())

        # Skip if the path is not obstacle-free
        if not is_obstacle_free(maze, q_near_state, q_new_state):
            continue

        parent = T[int(maze.index_from_state(q_near_state))]

        q_new = Node(
            parent=parent,
            index=int(maze.index_from_state(q_new_state)),
            state=q_new_state,
        )

        T[q_new.index] = q_new

        dist_to_goal = np.linalg.norm(np.array(maze.goal_state) - np.array(q_new.state))

        if dist_to_goal <= 1.0:
            pointer_node = deepcopy(q_new)
            path = [maze.state_from_index(pointer_node.index)]
            while pointer_node.index != maze.start_index:
                path.append(maze.state_from_index(pointer_node.parent.index))
                pointer_node = deepcopy(pointer_node.parent)
            print(f"Running time: {time.time()-start_time}")
            print(f"Path length: {len(path)}")
            maze.plot_path(path)
            print(path)
            return

        # pointer_node = deepcopy(q_new)
        # path = [maze.state_from_index(pointer_node.index)]
        # while pointer_node.index != maze.start_index:
        #     path.append(maze.state_from_index(pointer_node.parent.index))
        #     pointer_node = deepcopy(pointer_node.parent)

    # print(path)
    # maze.plot_path(path)


RRT(maze=maze_2)
# RRT(maze=maze_2)
