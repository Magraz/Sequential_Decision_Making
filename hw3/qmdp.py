import numpy as np
from maze import Maze

from utils import get_succ_states_and_trans
from qNavigate import value_iteration
from mostLikelySearch import belief_state_update
from pathlib import Path
import random


def extract_policy_from_q_table(world: Maze, q_table):
    policy = np.zeros(shape=(world.cols, world.rows))

    for col in range(world.cols):
        for row in range(world.rows):

            max_val = max(q_table[col, row])
            best_indices = [i for i, v in enumerate(q_table[col, row]) if v == max_val]

            policy[col, row] = random.choice(best_indices)

    return policy


if __name__ == "__main__":

    discount = 0.9
    noise = 0.2
    # 10% chance to move in a random direction
    # Python Maze class can also track the reward/discount (see below)
    maze = Maze(noise=noise, discount=discount)

    maze_filepath = Path(__file__).resolve().parent / "mazes/maze1.txt"
    maze.load_maze(maze_filepath)

    maze.reset()

    reward = 0
    cur_discount = discount

    q_table = np.zeros(shape=(maze.cols, maze.rows, len(maze.actions)))
    belief_state = np.ones(shape=(maze.cols, maze.rows)) * 1 / (maze.cols * maze.rows)

    state = maze.position

    value_table_mdp, _ = value_iteration(maze, discount)

    for _ in range(100):

        # Belief state update
        belief_state, act_belief_map = belief_state_update(maze, state, belief_state)

        # Update q table
        for action in maze.actions:
            s_states_trans_map = get_succ_states_and_trans(maze, state, action)

            for next_state, trans_prob in s_states_trans_map.items():

                q_table[state[0], state[1], action] += (
                    value_table_mdp[next_state[0], next_state[1]] * trans_prob
                )

            q_table[state[0], state[1], action] += maze.get_reward(state)

        act_val_map = {}
        for action in maze.actions:
            act_val_map[action] = 0
            s_states_trans_map = get_succ_states_and_trans(maze, state, action)
            # Update belief in actions
            for next_state, _ in s_states_trans_map.items():
                act_val_map[action] += (
                    q_table[next_state[0], next_state[1], action]
                    * belief_state[next_state[0], next_state[1]]
                )

        move = max(act_val_map, key=act_val_map.get)

        # try to move in the direction
        maze.step(int(move), move_target=True)

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

    # Report final reward
    print("Final Reward:", reward)
    print("Reward tracked by maze class:", maze.reward_current)

    policy = extract_policy_from_q_table(maze, q_table)
    maze.draw_maze(values=policy)
    input("Press Enter to continue...")
