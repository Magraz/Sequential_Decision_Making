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

goal = maze._get_target_location()


def value_iteration():
    epochs = 10000

    policy = np.zeros(shape=(maze.cols, maze.rows))
    value_table = np.zeros(shape=(maze.cols, maze.rows))
    new_value_table = value_table[:]

    # This shows how to use the step() function to move in the maze
    print("Using step function to move...")
    for epoch in range(epochs):

        for col in range(maze.cols):
            for row in range(maze.rows):
                state = (col, row)

                if not maze.get_observation(state):

                    new_value_table[col, row], best_action = value_update_rsas(
                        maze, discount, value_table, state
                    )
                    # policy[col, row] = best_action

        value_table = new_value_table[:]

    value_table[goal[0], goal[1]] = 1

    maze.draw_maze(values=value_table)
    input("Press Enter to continue...")
    return value_table, policy


value_table, _ = value_iteration()

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
