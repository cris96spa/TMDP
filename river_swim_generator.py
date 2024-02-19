import numpy as np

def generate_river(nS=6, small=5, large=10000):
    """
    Generate a river
    Args:
        nS (int): number of states
        gamma (float, optional): discount factor. Default to 1.
        hh
        seed (float, optional): pseudo-random generator seed. Default to None.

     Returns:
            (int, int, numpy.ndarray, numpy.ndarray, numpy.ndarray): number of states, 
                number of actions, probability matrix, reward matrix, initial state distribution
    """
    nA = 2
    p = compute_probabilities(nS, nA)
    r = compute_rewards(nS, nA, small, large)
    mu = compute_mu(nS)
    return nS, nA, p, r, mu

def compute_probabilities(nS, nA):
    """
    Compute probabilities
    Args:
        nS (int): number of states
        nA (int): number of actions

     Returns:
            (numpy.ndarray): probabilities of moving from each state to each other, when an action is taken [nS, nA, nS]
    """
    p = np.zeros((nS, nA, nS))
    for i in range(1, nS):
        # Set to 1 the probability of moving to the immediately left state, when left action is taken
        p[i, 0, i - 1] = 1

        # When not in the rightmost state, set the probability of moving left with right action to 0.1, while the probability of stay there to 0.6
        if i != nS - 1:
            p[i, 1, i - 1] = 0.1
            p[i, 1, i] = 0.6
        # When in the rightmost state, set the probability of moving left with right action to 0.7, while the probability of stay there to 0.3
        else:
            p[i, 1, i - 1] = 0.7
            p[i, 1, i] = 0.3
    # When in middle states, set the probability
    for i in range(nS - 1):
        p[i, 1, i + 1] = 0.3
    # state 0
    p[0, 0, 0] = 1
    p[0, 1, 0] = 0.7

    return p

def compute_rewards(nS, nA, small, large):
    """
    Compute rewards

    Args:
        nS (int): number of states
        nA (int): number of actions
        small (int, optional): small reward. Defaults to 5.
        large (int, optional): large reward. Defaults to 10000.

     Returns:
            (numpy.ndarray): immediate reward matrix of moving from each state to each other when an action is taken [nS, nA, nS]
    """
    # initialize all rewards to 0
    r = np.zeros((nS, nA, nS))
    
    # set to small the reward associated to the left action on the leftmost state, when remaining there
    r[0, 0, 0] = small
    
    # set to large the reward associated to the right action in the rightmost state, when remaining there
    r[nS - 1, 1, nS - 1] = large
    return r


"""
    Compute initial state probability vector
        @ns: number of states
        return the initial state probability vector
"""
def compute_mu(nS):
    # Uniform probability distribution of initial states
    mu = np.ones(nS)/nS
    return mu
