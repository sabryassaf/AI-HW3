from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float = 10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    U_final = np.array(U_init, dtype=np.float64)
    gamma = mdp.gamma

    while True:
        u_old = U_final.copy()
        maximal_change = 0
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                # fetch the state
                state = (row, col)
                if mdp.board[row][col] == 'WALL':
                    continue
                if state in mdp.terminal_states:
                    U_final[row, col] = float(mdp.get_reward(state))
                    continue
                # find the best action to take from the current state
                max_utility = float('-inf')
                reward = float(mdp.get_reward(state))

                for action in mdp.actions:
                    expected_utility = 0
                    if action not in mdp.transition_function:
                        continue
                    for prob, current_action_with_prob in zip(mdp.transition_function[action], mdp.actions):
                        next_state = mdp.step(state, current_action_with_prob)
                        expected_utility += prob * u_old[next_state[0], next_state[1]]

                    expected_utility = reward + gamma * expected_utility

                    if expected_utility > max_utility:
                        max_utility = expected_utility

                    # handle inf
                    if max_utility == float('-inf'):
                        max_utility = 0

                # update the utility of the current state
                U_final[row, col] = max_utility

                # update the maximal change
                maximal_change = max(maximal_change, abs(U_final[row, col] - u_old[row, col]))

        # check if the maximal change is smaller than epsilon
        if maximal_change < epsilon:
            break

    return U_final


def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Initialize an empty policy array
    policy = np.empty((mdp.num_row, mdp.num_col), dtype=object)

    # Iterate over all states
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            state = (row, col)

            # Skip terminal states and walls
            if state in mdp.terminal_states or mdp.board[row][col] == 'WALL':
                policy[row, col] = None
                continue

            # Determine the best action for this state
            best_action = None
            max_utility = float('-inf')

            for action in mdp.actions:
                expected_utility = 0

                # Use transition probabilities and utilities to calculate expected utility
                for prob, current_action_with_prob in zip(mdp.transition_function[action], mdp.actions):
                    next_state = mdp.step(state, current_action_with_prob)
                    expected_utility += prob * U[next_state[0], next_state[1]]

                # Compare to find the best action
                if expected_utility > max_utility:
                    max_utility = expected_utility
                    best_action = action

            # Assign the best action to the policy
            policy[row, col] = best_action

    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    number_of_states = mdp.num_row * mdp.num_col
    P = np.zeros((number_of_states, number_of_states))
    R = np.zeros(number_of_states)
    U = np.zeros(number_of_states)
    I = np.eye(number_of_states, number_of_states)
    gamma = mdp.gamma

    policy = np.array(policy)
    # begin filling the P and R matrices
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            state = (row, col)

            if state in mdp.terminal_states or mdp.board[row][col] == 'WALL':
                continue
            action = Action(policy[row, col])
            # change the action to the actual action

            transition_probs = mdp.transition_function[action]

            reward = float(mdp.get_reward(state))
            state_index = row * mdp.num_col + col
            R[state_index] = reward

            for prob, current_action_with_prob in zip(mdp.transition_function[action], mdp.actions):
                next_state = mdp.step(state, current_action_with_prob)
                next_state_index = next_state[0] * mdp.num_col + next_state[1]
                P[state_index, next_state_index] += prob

    # calculate the utility of each state
    U = np.linalg.solve(I - gamma * P, R)
    U = U.reshape((mdp.num_row, mdp.num_col))

    return U


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    num_of_states = mdp.num_row * mdp.num_col
    U = np.zeros(num_of_states)
    policy_init = np.array(policy_init)
    optimal_policy = policy_init.copy()

    # iterate over the policy evaluation and policy improvement steps until convergence
    while True:
        U = policy_evaluation(mdp, optimal_policy)
        # convergence status
        policy_stable = True

        for state in range(num_of_states):
            row, col = divmod(state, mdp.num_col)
            state = (row, col)

            # check state status
            if state in mdp.terminal_states or mdp.board[row][col] == 'WALL':
                optimal_policy[row, col] = None
                continue

            # fetch current policy
            old_action = Action(optimal_policy[state])
            new_action = None
            max_utility = float('-inf')
            reward = float(mdp.get_reward(state))

            # iterate over all possible actions
            for action in mdp.actions:
                expected_utility = 0
                if action not in mdp.transition_function:
                    continue

                for prob, current_action_with_prob in zip(mdp.transition_function[action], mdp.actions):
                    next_state = mdp.step(state, current_action_with_prob)
                    expected_utility += prob * U[next_state[0], next_state[1]]


                expected_utility = reward + mdp.gamma * expected_utility
                if expected_utility > max_utility:
                    max_utility = expected_utility
                    new_action = action

            # check for change
            if old_action != new_action:
                policy_stable = False
                optimal_policy[state] = new_action

        # check for convergence
        if policy_stable:
            break

    return optimal_policy


def adp_algorithm(
        sim: Simulator,
        num_episodes: int,
        num_rows: int = 3,
        num_cols: int = 4,
        actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """

    # initiate transition_probs as a dictionary of dictionaries, each key in the outer dictionary indicates an action, and its value is
    # a dictionary of all possible actions with their count
    transition_probs = {action: {possible_action: 0 for possible_action in actions} for action in actions}

    # keep track of chosen actions
    chosen_actions = {action: 0 for action in actions}

    # initiate the reward matrix
    reward_matrix = np.zeros((num_rows, num_cols))

    # iterate over the episodes
    for episode in sim.replay(num_episodes):
        for step in episode:
            state, reward, chosen_action, actual_action = step
            # pass invalid actions
            if chosen_action is None or actual_action is None:
                continue

            # increase the total count of the chosen action
            chosen_actions[chosen_action] += 1

            # increase the count of the actual action given the chosen action
            transition_probs[chosen_action][actual_action] += 1

            # update the reward if it was zero
            if reward_matrix[state[0], state[1]] == 0:
                reward_matrix[state[0], state[1]] = reward

    # calculate the transition probabilities
    conditional_probs = {
        chosen_action: {
            actual_action: (transition_probs[chosen_action][actual_action] / chosen_actions[chosen_action])
            if chosen_actions[chosen_action] > 0 else 0.0
            for actual_action in actions
        }
        for chosen_action in actions
    }

    return reward_matrix, conditional_probs
