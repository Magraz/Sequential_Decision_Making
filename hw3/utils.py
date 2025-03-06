from maze import Maze
import numpy as np


def value_update_rsas(world: Maze, gamma: float, value_table, curr_state):

    q_values = []

    for action in world.actions:

        value_sum = 0

        succesor_states = get_succ_states_and_trans(
            world=world, state=curr_state, action=action
        )

        for next_state, trans_prob in succesor_states.items():
            # if (world.is_move_valid(action) is not None) and world.is_position_valid(
            #     next_state
            # ):
            if world.is_position_valid(next_state):
                reward = world.get_reward(next_state)
                next_state_value = trans_prob * (
                    reward + gamma * value_table[next_state[0], next_state[1]]
                )
                value_sum += next_state_value

        q_values.append(value_sum)

    q_values = np.array(q_values)

    return np.max(q_values), np.argmax(q_values)


def get_succ_states_and_trans(world: Maze, state, action):

    action_next_state_map = get_succ_states(world, state)

    sp = 1 - world.noise
    success = sp + (1 - sp) / 4
    noise = (1 - sp) / 4

    p = {
        0: [success, noise, noise, noise],
        1: [noise, success, noise, noise],
        2: [noise, noise, success, noise],
        3: [noise, noise, noise, success],
    }
    succesor_states_map = {}

    for action_idx, s_state in action_next_state_map.items():
        if s_state is not None:
            succesor_states_map[s_state] = p[action][action_idx]

    return succesor_states_map


def get_succ_states(world: Maze, state):

    action_next_state_map = {}

    for action in world.actions:
        action_next_state_map[action] = world.is_move_valid(action, position=state)

    return action_next_state_map


def extract_policy(world: Maze, value_table):
    policy = np.zeros_like(value_table)
    for col in range(world.cols):
        for row in range(world.rows):
            state = (col, row)
            act_next_state_map = get_succ_states(world, state)
            act_val_map = {}

            for action, s_state in act_next_state_map.items():
                if s_state is not None:
                    act_val_map[action] = value_table[s_state[0], s_state[1]]

            if act_val_map:
                policy[col, row] = max(act_val_map, key=act_val_map.get)

    return policy
