"""
random_explorer.py

Explores the world purely randomly
This file also demonstrates how you can use the provided
Maze class.
"""

import numpy as np
from maze import Maze
from utils import get_succ_states
from pathlib import Path


def belief_state_update(world: Maze, state, belief_state):
    act_belief_map = {}
    act_next_state_map = get_succ_states(world, state)
    valid_actions = len({k: v for k, v in act_next_state_map.items() if v is not None})

    for action, s_state in act_next_state_map.items():
        if s_state is not None:
            belief_state[s_state[0], s_state[1]] += (
                belief_state[state[0], state[1]] / valid_actions
            )
            act_belief_map[action] = belief_state[s_state[0], s_state[1]]

    if not world.get_observation(world.position):
        belief_state[state[0], state[1]] = 0

    return belief_state, act_belief_map


if __name__ == "__main__":
    discount = 0.9
    noise = 0.3
    # 10% chance to move in a random direction
    # Python Maze class can also track the reward/discount (see below)
    maze = Maze(noise=noise, discount=discount)

    maze_filepath = Path(__file__).resolve().parent / "mazes/maze1.txt"
    maze.load_maze(maze_filepath)

    # You can reset the maze with reset()
    maze.reset()

    reward = 0
    cur_discount = discount

    belief_state = np.ones(shape=(maze.cols, maze.rows)) * 1 / (maze.cols * maze.rows)

    state = maze.position

    for _ in range(100):
        belief_state, act_belief_map = belief_state_update(maze, state, belief_state)

        most_likely_action = max(act_belief_map, key=act_belief_map.get)

        maze.step(most_likely_action, move_target=True)

        state = maze.position

        # Update the reward
        reward += maze.get_reward() * cur_discount
        # Update the discount factor
        cur_discount *= discount

        # Check to see if you are at the goal
        goal_info = "Not At Goal"
        if maze.get_observation(maze.position):
            goal_info = "At Goal"
            belief_state[state[0], state[1]] = 1

        # Print out location of the agent
        print(f"{maze.position}: {goal_info}")

    print(belief_state)
    # Report final reward
    print("Final Reward:", reward)
    print("Reward tracked by maze class:", maze.reward_current)

    maze.draw_maze(values=belief_state)
    input("Press Enter to continue...")
