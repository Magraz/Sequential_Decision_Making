from provided_code.maze import *
from provided_code.priority_queue import *
import numpy as np
import os


def heuristic(current, goal, method="manhattan"):
    x1, y1 = current
    x2, y2 = goal
    if method == "manhattan":
        return abs(x1 - x2) + abs(y1 - y2)
    elif method == "euclidean":
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return 0


def time_heuristic(current, goal, method="manhattan"):
    x1, y1, vx1, vy1 = current  # current state
    x2, y2, _, _ = goal
    if method == "manhattan":
        distance = abs(x1 - x2) + abs(y1 - y2)
    elif method == "euclidean":
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    avg_velocity = (
        max(abs(vx1), abs(vy1)) if vx1 or vy1 else 1
    )  # average velocity of the current state

    return distance / avg_velocity  # estimated time to goal


def a_star(maze, heuristic_method="manhattan"):
    # initilize start and goal
    start = maze.get_start()
    goal = maze.get_goal()

    # initilize open and closed sets
    O = PriorityQueue()  # open set
    C = set()  # closed set

    O.insert(start, priority=0)  # add start to open set

    parent = {}  # parent of nodes
    g = {start: 0}  # cost to get to node

    while len(O) > 0:  # while open set is not empty
        n_best = O.pop()  # get the node with the lowest f(n)
        C.add(n_best)  # add n_best to closed set

        if n_best == goal:  # if goal is reached
            # reconstruct the path
            path = []  # initilize path
            while n_best in parent:  # while there is a parent
                path.append(maze.state_from_index(n_best))  # add the state to the path
                n_best = parent[n_best]  # move to the parent
            path.append(maze.state_from_index(start))  # add the start to the path
            path.reverse()  # reverse the path
            return path

        # expand neighbors of n_best
        for neighbor in maze.get_neighbors(n_best):  # for each neighbor
            if neighbor in C:  # if neighbor is in closed set
                continue  # skip the neighbor

            # calculate the cost to get to the neighbor
            tentative_g = g[n_best] + 1  # cost to get to neighbor is 1

            if not O.test(neighbor):  # if neighbor is not in open set
                O.insert(
                    neighbor, priority=float("inf")
                )  # add neighbor to open set with priority infinity

            # check if the cost to get to the neighbor is less than the current cost
            if tentative_g < g.get(neighbor, float("inf")):  # if the cost is less
                parent[neighbor] = n_best  # set the parent of the neighbor
                g[neighbor] = tentative_g  # set the cost to get to the neighbor
                f = tentative_g + heuristic(  # calculate f(n)
                    maze.state_from_index(neighbor),  # get the state from the index
                    maze.state_from_index(goal),  # get the goal state
                    method=heuristic_method,  # use the heuristic method
                )
                O.insert(
                    neighbor, priority=f
                )  # add the neighbor to the open set with priority f

    return None  # return None if no path is found


def a_star_epsilon(maze, epsilon, heuristic_method="manhattan"):
    # initilize start and goal
    start = maze.get_start()
    goal = maze.get_goal()

    # initilize open and closed sets
    O = PriorityQueue()  # open set
    C = set()  # closed set

    O.insert(start, priority=0)  # add start to open set

    parent = {}  # parent of nodes
    g = {start: 0}  # cost to get to node

    nodes_expanded = 0  # initilize nodes expanded

    while len(O) > 0:  # while open set is not empty
        n_best = O.pop()  # get the node with the lowest f(n)
        C.add(n_best)  # add n_best to closed set
        nodes_expanded += 1  # increment nodes expanded

        if n_best == goal:  # if goal is reached
            # reconstruct the path
            path = []  # initilize path
            while n_best in parent:  # while there is a parent
                path.append(maze.state_from_index(n_best))  # add the state to the path
                n_best = parent[n_best]  # move to the parent
            path.append(maze.state_from_index(start))  # add the start to the path
            path.reverse()  # reverse the path
            return path, nodes_expanded

        # expand neighbors of n_best
        for neighbor in maze.get_neighbors(n_best):  # for each neighbor
            if neighbor in C:  # if neighbor is in closed set
                continue  # skip the neighbor

            # calculate the cost to get to the neighbor
            tentative_g = g[n_best] + 1  # cost to get to neighbor is 1

            if not O.test(neighbor):  # if neighbor is not in open set
                O.insert(
                    neighbor, priority=float("inf")
                )  # add neighbor to open set with priority infinity

            # check if the cost to get to the neighbor is less than the current cost
            if tentative_g < g.get(neighbor, float("inf")):  # if the cost is less
                parent[neighbor] = n_best  # set the parent of the neighbor
                g[neighbor] = tentative_g  # set the cost to get to the neighbor
                f = tentative_g + epsilon * heuristic(  # calculate f(n)
                    maze.state_from_index(neighbor),  # get the state from the index
                    maze.state_from_index(goal),  # get the goal state
                    method=heuristic_method,  # use the heuristic method
                )
                O.insert(
                    neighbor, priority=f
                )  # add the neighbor to the open set with priority f
    # #print the length of the closed set
    #     print("Length of closed set: ", end = " ")
    #     print(len(C))
    return None, nodes_expanded  # return None if no path is found


def a_star_4D(maze, heuristic_method="manhattan"):
    # initilize start and goal
    start = maze.get_start()
    goal = maze.get_goal()

    # initilize open and closed sets
    O = PriorityQueue()  # open set
    C = set()  # closed set

    O.insert(start, priority=0)  # add start to open set

    parent = {}  # parent of nodes
    g = {start: 0}  # cost to get to node

    while len(O) > 0:  # while open set is not empty
        n_best = O.pop()  # get the node with the lowest f(n)
        C.add(n_best)  # add n_best to closed set

        if n_best == goal:  # if goal is reached
            # reconstruct the path
            path = []  # initilize path
            while n_best in parent:  # while there is a parent
                path.append(maze.state_from_index(n_best))  # add the state to the path
                n_best = parent[n_best]  # move to the parent
            path.append(maze.state_from_index(start))  # add the start to the path
            path.reverse()  # reverse the path
            return path

        # expand neighbors of n_best
        for neighbor in maze.get_neighbors(n_best):  # for each neighbor
            if neighbor in C:  # if neighbor is in closed set
                continue  # skip the neighbor

            neighbor_state = maze.state_from_index(
                neighbor
            )  # get the state of the neighbor to get the velocity
            vx, vy = neighbor_state[2:]  # get the velocity of the neighbor

            time_cost = 1 / max(
                1, abs(vx) + abs(vy)
            )  # calculate the time cost based on velcocity
            # calculate the cost to get to the neighbor with time cost
            tentative_g = g[n_best] + time_cost

            if not O.test(neighbor):  # if neighbor is not in open set
                O.insert(
                    neighbor, priority=float("inf")
                )  # add neighbor to open set with priority infinity

            # check if the cost to get to the neighbor is less than the current cost
            if tentative_g < g.get(neighbor, float("inf")):  # if the cost is less
                parent[neighbor] = n_best  # set the parent of the neighbor
                g[neighbor] = tentative_g  # set the cost to get to the neighbor
                f = (
                    tentative_g
                    + time_heuristic(  # calculate f(n) based on time heuristic
                        maze.state_from_index(neighbor),  # get the state from the index
                        maze.state_from_index(goal),  # get the goal state
                        method=heuristic_method,  # use the heuristic method
                    )
                )
                O.insert(
                    neighbor, priority=f
                )  # add the neighbor to the open set with priority f

    return None  # return None if no path is found


def a_star_4D_epsilon(maze, epsilon, heuristic_method="manhattan"):
    # initilize start and goal
    start = maze.get_start()
    goal = maze.get_goal()

    # initilize open and closed sets
    O = PriorityQueue()  # open set
    C = set()  # closed set

    O.insert(start, priority=0)  # add start to open set

    parent = {}  # parent of nodes
    g = {start: 0}  # cost to get to node

    nodes_expanded = 0  # initilize nodes expanded

    while len(O) > 0:  # while open set is not empty
        n_best = O.pop()  # get the node with the lowest f(n)
        C.add(n_best)  # add n_best to closed set
        nodes_expanded += 1  # increment nodes expanded

        if n_best == goal:  # if goal is reached
            # reconstruct the path
            path = []  # initilize path
            while n_best in parent:  # while there is a parent
                path.append(maze.state_from_index(n_best))  # add the state to the path
                n_best = parent[n_best]  # move to the parent
            path.append(maze.state_from_index(start))  # add the start to the path
            path.reverse()  # reverse the path
            return path, nodes_expanded

        # expand neighbors of n_best
        for neighbor in maze.get_neighbors(n_best):  # for each neighbor
            if neighbor in C:  # if neighbor is in closed set
                continue  # skip the neighbor

            neighbor_state = maze.state_from_index(
                neighbor
            )  # get the state of the neighbor to get the velocity
            vx, vy = neighbor_state[2:]  # get the velocity of the neighbor

            time_cost = 1 / max(
                1, abs(vx) + abs(vy)
            )  # calculate the time cost based on velcocity
            # calculate the cost to get to the neighbor with time cost
            tentative_g = g[n_best] + time_cost

            if not O.test(neighbor):  # if neighbor is not in open set
                O.insert(
                    neighbor, priority=float("inf")
                )  # add neighbor to open set with priority infinity

            # check if the cost to get to the neighbor is less than the current cost
            if tentative_g < g.get(neighbor, float("inf")):  # if the cost is less
                parent[neighbor] = n_best  # set the parent of the neighbor
                g[neighbor] = tentative_g  # set the cost to get to the neighbor
                f = (
                    tentative_g
                    + epsilon
                    * time_heuristic(  # calculate f(n) based on time heuristic
                        maze.state_from_index(neighbor),  # get the state from the index
                        maze.state_from_index(goal),  # get the goal state
                        method=heuristic_method,  # use the heuristic method
                    )
                )
                O.insert(
                    neighbor, priority=f
                )  # add the neighbor to the open set with priority f

    return None, nodes_expanded  # return None if no path is found


if __name__ == "__main__":

    # import maze from pgm file
    dirname = os.path.dirname(__file__)

    maze1_filename = os.path.join(dirname, "provided_code/maze1.pgm")

    maze2_filename = os.path.join(dirname, "provided_code/maze2.pgm")

    maze = Maze2D.from_pgm(maze2_filename)

    # run a* search
    path = a_star(maze, heuristic_method="manhattan")

    # path = a_star_epsilon(maze, 1, heuristic_method="manhattan")

    if path:
        maze.plot_path(path, "A* Search - Implementation")
    else:
        print("No path found.")
