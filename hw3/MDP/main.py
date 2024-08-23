from mdp_rl_implementation import value_iteration, get_policy, policy_evaluation, policy_iteration, adp_algorithm
from mdp import MDP
from simulator import Simulator
from mdp import Action


def example_driver():
    """
    This is an example of a driver function, after implementing the functions
    in "mdp_rl_implementation.py" you will be able to run this code with no errors.
    """

    mdp = MDP.load_mdp()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@ The board and rewards @@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    mdp.print_rewards()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Value iteration @@@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    U = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    print("\nInitial utility:")
    mdp.print_utility(U)
    print("\nFinal utility:")
    U_new = value_iteration(mdp, U)
    mdp.print_utility(U_new)
    print("\nFinal policy:")
    policy = get_policy(mdp, U_new)
    mdp.print_policy(policy)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Policy iteration @@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print("\nPolicy evaluation:")
    U_eval = policy_evaluation(mdp, policy)
    mdp.print_utility(U_eval)

    policy = [['UP', 'UP', 'UP', 0],
              ['UP', 'WALL', 'UP', 0],
              ['UP', 'UP', 'UP', 'UP']]

    print("\nInitial policy:")
    mdp.print_policy(policy)
    print("\nFinal policy:")
    policy_new = policy_iteration(mdp, policy)
    mdp.print_policy(policy_new)

    print("Done!")


def adp_example_driver():
    sim = Simulator()

    reward_matrix, transition_probabilities = adp_algorithm(sim, num_episodes=10)

    print("Reward Matrix:")
    print(reward_matrix)

    print("\n Transition Probabilities:")
    for action, probs in transition_probabilities.items():
        print(f"{action}: {probs}")


def format_transition_probabilities(transition_probabilities):
    """
    Format the transition probabilities using Action enum instances as keys.
    """
    formatted_transition_probabilities = {}
    actions = [Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]

    for action, probs in transition_probabilities.items():
        formatted_probs = tuple([probs.get(action, 0) for action in actions])
        formatted_transition_probabilities[action] = formatted_probs

    return formatted_transition_probabilities


def convert_policy_to_action_enum(policy):
    """
    Convert a policy represented as a list of action names to a list of Action enum instances.
    """
    action_mapping = {
        'UP': Action.UP,
        'DOWN': Action.DOWN,
        'RIGHT': Action.RIGHT,
        'LEFT': Action.LEFT,
        'WALL': None,  # Assuming 'WALL' is used for walls and should be kept as None
        0: None        # Assuming 0 is used for terminal states and should be kept as None
    }
    converted_policy = []
    for row in policy:
        converted_row = [action_mapping[action] if action in action_mapping else action for action in row]
        converted_policy.append(converted_row)

    return converted_policy


def adp_extra_examples():
    sim = Simulator()
    num_episodes_list = [10, 100, 1000]
    for num_episodes in num_episodes_list:
        reward_matrix, transition_probabilities = adp_algorithm(sim, num_episodes=num_episodes)
        # Convert reward matrix to a format that can be passed to MDP
        reward_matrix_list = []
        for row in reward_matrix:
            reward_matrix_list.append([str(element) for element in row])
        # Format the transition probabilities using Action enum instances as keys
        formatted_transition_probabilities = format_transition_probabilities(transition_probabilities)
        terminal_states = [(0, 3), (1, 3)]
        # add the expected wall
        reward_matrix_list[1][1] = 'WALL'
        gamma = 0.9
        # Create the MDP instance with Action enums as transition function keys
        current_mdp = MDP(
            board=reward_matrix_list,
            terminal_states=terminal_states,
            transition_function=formatted_transition_probabilities,
            gamma=gamma
        )
        # Initial policy for policy iteration (using action names as strings)
        policy = [['DOWN', 'DOWN', 'DOWN', 0],
                  ['DOWN', 0, 'DOWN', 0],
                  ['DOWN', 'DOWN', 'DOWN', 'DOWN']]
        # Convert the policy to use Action enum instances
        actions_policy = convert_policy_to_action_enum(policy)
        # Perform policy iteration on the updated MDP
        policy_new = policy_iteration(current_mdp, actions_policy)
        # Print the resulting policy
        print(f"\nPolicy after {num_episodes} episodes:")
        current_mdp.print_policy(policy_new)


if __name__ == '__main__':
    # run our example
    example_driver()
    # adp_example_driver()
    # adp_extra_examples()
