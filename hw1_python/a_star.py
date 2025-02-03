from provided_code.maze import Maze2D, Maze4D, Maze
from provided_code.priority_queue import PriorityQueue
import os
import numpy as np
from copy import deepcopy
import time

dirname = os.path.dirname(__file__)

maze1_filename = os.path.join(dirname, "provided_code/maze1.pgm")

maze2_filename = os.path.join(dirname, "provided_code/maze2.pgm")


maze_1 = Maze2D.from_pgm(maze1_filename)

# maze_1.start_state = (1, 1)
# maze_1.goal_state = (maze_1.cols, maze_1.rows)

# print(f"Maze 1")
# print(f"Goal State {maze_1.goal_state}")
# print(f"Goal Index {maze_1.index_from_state(maze_1.goal_state)}")

maze_2 = Maze2D.from_pgm(maze2_filename)

# maze_2.start_state = (1, 1)
# maze_2.goal_state = (maze_2.cols, maze_2.rows)

# print(f"Maze 2")
# print(f"Goal State {maze_2.goal_state}")
# print(f"Goal Index {maze_2.index_from_state(maze_2.goal_state)}")


def heuristic(neighbor_state, goal_state, type: str):
    h = 0
    match (type):
        case "euclidean":
            h = np.linalg.norm(np.array(neighbor_state) - np.array(goal_state))
        case "manhatan":
            h = np.sum(np.abs(np.array(neighbor_state) - np.array(goal_state)))

    return h


class Node:
    def __init__(
        self,
        parent,
        index: int,
        state: tuple,
        h_cost: float = 0.0,
        g_cost: float = 0.0,
        f_cost: float = 0.0,
    ):
        self.parent = parent
        self.index = index
        self.state = state
        self.h_cost = h_cost
        self.g_cost = g_cost
        self.f_cost = f_cost


def A_Star(
    maze: Maze2D | Maze4D,
):
    open_list = {}
    closed_list = set()

    start_node = Node(parent=None, index=maze.start_index, state=maze.start_state)
    start_node.h_cost = np.linalg.norm(
        np.array(start_node.state) - np.array(maze.goal_state)
    )
    start_node.f_cost = start_node.g_cost + start_node.h_cost

    open_list[start_node.index] = start_node

    while len(open_list) > 0:
        node_idx_with_lowest_f_cost = min(open_list, key=lambda k: open_list[k].f_cost)
        current_node = open_list.pop(node_idx_with_lowest_f_cost)

        # Add to closed list
        closed_list.add(current_node.index)

        # Get path
        if current_node.index == maze.goal_index:
            path = []
            while current_node:
                path.append(maze.state_from_index(current_node.index))
                current_node = current_node.parent
            path.reverse()
            print(path)
            maze.plot_path(path=path)
            return

        for neighbor_idx in maze.get_neighbors(current_node.index):

            if neighbor_idx in closed_list:
                continue

            neighbor = Node(
                parent=current_node,
                index=neighbor_idx,
                state=maze.state_from_index(neighbor_idx),
            )

            dist_to_neighbor = np.linalg.norm(
                np.array(neighbor.state) - np.array(current_node.state)
            )

            # Compute costs
            tentative_g_cost = current_node.g_cost + dist_to_neighbor
            h = heuristic(neighbor.state, maze.goal_state, "manhattan")

            # If neighbor is already in open_list with better g_cost, skip
            if (
                neighbor_idx in open_list
                and tentative_g_cost >= open_list[neighbor_idx].g_cost
            ):
                continue

            # Update costs and add to open list
            neighbor.g_cost = tentative_g_cost
            neighbor.h_cost = h
            neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
            open_list[neighbor.index] = neighbor  # Update open list


def A_Star_Epsilon(
    maze: Maze2D | Maze4D, epsilon: float = 10, runtime_limit: float = 1.0
):
    start_time = time.time()

    while True:

        open_list = {}
        closed_list = set()

        start_node = Node(parent=None, index=maze.start_index, state=maze.start_state)
        start_node.h_cost = np.linalg.norm(
            np.array(start_node.state) - np.array(maze.goal_state)
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost

        open_list[start_node.index] = start_node

        expanded_nodes = 0

        while len(open_list) > 0:
            node_idx_with_lowest_f_cost = min(
                open_list, key=lambda k: open_list[k].f_cost
            )
            current_node = open_list.pop(node_idx_with_lowest_f_cost)
            expanded_nodes += 1

            # Add to closed list
            closed_list.add(current_node.index)

            # Get path
            path = []
            if current_node.index == maze.goal_index:
                pointer_node = deepcopy(current_node)
                path = [maze.state_from_index(pointer_node.index)]

                while pointer_node.index != maze.start_index:
                    path.append(maze.state_from_index(pointer_node.parent.index))
                    pointer_node = deepcopy(pointer_node.parent)

                # Print data
                print("\n")
                print(f"Runtime: {(time.time()-start_time)}")
                print(f"Epsilon: {epsilon}")
                print(f"Path Length: {len(path)}")
                print(f"Expanded nodes: {expanded_nodes}")

                if epsilon == 1 or ((time.time() - start_time) > runtime_limit):
                    return

                # Update epsilon
                new_epsilon = epsilon - 0.5 * (epsilon - 1)

                if new_epsilon < 1.001:
                    new_epsilon = 1

                epsilon = new_epsilon

                break

            elif ((time.time() - start_time) > runtime_limit) or (epsilon == 1):
                # Print data
                print("\n")
                print(f"Runtime: {(time.time()-start_time)}")
                print(f"Epsilon: {epsilon}")
                print(f"Path Length: {len(path)}")
                print(f"Expanded nodes: {expanded_nodes}")
                return

            for neighbor_idx in maze.get_neighbors(current_node.index):

                if neighbor_idx in closed_list:
                    continue

                neighbor = Node(
                    parent=current_node,
                    index=neighbor_idx,
                    state=maze.state_from_index(neighbor_idx),
                )

                dist_to_neighbor = np.linalg.norm(
                    np.array(neighbor.state) - np.array(current_node.state)
                )

                # Compute costs
                tentative_g_cost = current_node.g_cost + dist_to_neighbor
                h = heuristic(neighbor.state, maze.goal_state, "manhattan")

                # If neighbor is already in open_list with better g_cost, skip
                if (
                    neighbor_idx in open_list
                    and tentative_g_cost >= open_list[neighbor_idx].g_cost
                ):
                    continue

                # Update costs and add to open list
                neighbor.g_cost = tentative_g_cost
                neighbor.h_cost = h * epsilon
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                open_list[neighbor.index] = neighbor  # Update open list


# A_Star(maze=maze_2)
# A_Star(maze=maze_1)

A_Star_Epsilon(maze=maze_2, runtime_limit=1.0)
