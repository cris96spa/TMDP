import numpy as np

"""
    Generate a river swim 
    Args:
        size (int): number of states
        small (float): small reward
        large (float): large reward

    Returns:    
        (int, int, numpy.ndarray, numpy.ndarray, numpy.ndarray): number of states, 
                number of actions, probability matrix, reward matrix, initial state distribution
"""
def generate_river(size=6, small=5, large=10000):

    nA = 2
    p = compute_probabilities(size, nA)
    r = compute_rewards(size, nA, small, large)
    return size, nA, p, r

"""
    Compute probabilities as follows:
        - 1.0 to loop for left action in leftmost state 
        - 0.7 to loop for right action in the leftmost state

        - 1.0 to go left for left action in any non-first state

        - 0.1 to go left for right action in any middle state
        - 0.6 to loop for right action in any middle state
        - 0.3 to go right for right action in any middle state
        
        - 0.3 to loop for right action in the rightmost state
        - 0.7 to go left for right action in the rightmost state

    Args:
        size (int): number of states
        nA (int): number of actions

     Returns:
            (numpy.ndarray): probabilities of moving from each state to each other, when an action is taken [size, nA, size]
"""
def compute_probabilities(size, nA):
 
    p = np.zeros((size, nA, size))
    for i in range(1, size):
        # Set to 1 the probability of moving to the immediately left state, when left action is taken
        p[i, 0, i - 1] = 1

        # When not in the rightmost state, set the probability of moving left with right action to 0.1, while the probability of stay there to 0.6
        if i != size - 1:
            p[i, 1, i - 1] = 0.1
            p[i, 1, i] = 0.6
        # When in the rightmost state, set the probability of moving left with right action to 0.7, while the probability of stay there to 0.3
        else:
            p[i, 1, i - 1] = 0.7
            p[i, 1, i] = 0.3
    # When in middle states, set the probability
    for i in range(size - 1):
        p[i, 1, i + 1] = 0.3
    # state 0
    p[0, 0, 0] = 1
    p[0, 1, 0] = 0.7

    return p

"""
    Compute rewards

    Args:
        size (int): number of states
        nA (int): number of actions
        small (int, optional): small reward. Defaults to 5.
        large (int, optional): large reward. Defaults to 10000.

     Returns:
            (numpy.ndarray): immediate reward matrix of moving from each state to each other when an action is taken [size, nA, size]
"""
def compute_rewards(size, nA, small, large):

    # initialize all rewards to 0
    r = np.zeros((size, nA, size))
    
    # set to small the reward associated to the left action on the leftmost state, when remaining there
    r[0, 0, 0] = small
    
    # set to large the reward associated to the right action in the rightmost state, when remaining there
    r[size - 1, 1, size - 1] = large
    return r

"""
    Compute rewards

    Args:
        size (int): number of states

     Returns:
            (numpy.ndarray): initial state probability vector
"""
