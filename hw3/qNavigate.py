"""
random_explorer.py

Explores the world purely randomly
This file also demonstrates how you can use the provided
Maze class.
"""

import numpy as np
from maze import Maze
from utils import value_update_rsas, extract_policy
from pathlib import Path

discount = 0.9
noise = 0.2
# 10% chance to move in a random direction
# Python Maze class can also track the reward/discount (see below)
maze = Maze(noise=noise, discount=discount)

maze_filepath = Path(__file__).resolve().parent / "mazes/maze0.txt"
maze.load_maze(maze_filepath)

# You can reset the maze with reset()
maze.reset()


def value_iteration(world: Maze, discount: float):
    epochs = 1000

    goal = world._get_target_location()

    policy = np.zeros(shape=(world.cols, world.rows))
    value_table = np.zeros(shape=(world.cols, world.rows))
    new_value_table = value_table[:]

    for epoch in range(epochs):

        for col in range(world.cols):
            for row in range(world.rows):
                state = (col, row)

                if not world.get_observation(state):

                    new_value_table[col, row], _ = value_update_rsas(
                        world, discount, value_table, state
                    )
                    # policy[col, row] = best_action

        value_table = new_value_table[:]

    value_table[goal[0], goal[1]] = 1

    return value_table, policy


if __name__ == "__main__":

    value_table, _ = value_iteration(maze, discount)

    policy = extract_policy(maze, value_table)

    maze.reset()

    reward = 0
    cur_discount = discount

    # This shows how to use the step() function to move in the maze
    print("Using step function to move...")
    for _ in range(100):
        # choose a random direction to move in
        state = maze.position
        move = policy[state[0], state[1]]
        # try to move in the direction
        maze.step(int(move))

        # Update the reward
        reward += maze.get_reward() * cur_discount

        # Update the discount factor
        cur_discount *= discount

    # Report final reward
    print("Final Reward:", reward)
    print("Reward tracked by maze class:", maze.reward_current)

    maze.draw_maze(values=policy)
    input("Press Enter to continue...")
