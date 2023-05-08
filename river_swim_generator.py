import numpy as np

"""
    A river swim environment is an environment in which there are several sequential states and
    only two possible moves, left and right. It is assumed that left move is in the same direction of the river
    flow, hence it always lead to the immediatly left state, whereas the right move is done against the flow, 
    meaning that it has a very small probability of leading to the left, a pretty high probability of remaining 
    in the same state and a small probability of moving to the right. The right you are able to move, the higher
    will be rewards.
"""
"""
    Generate a river
        @n: number of states
        @small: small reward
        @large: large reward
        return a river made up of n states, 2 actions, p transition dynamic, r rewards and mu initial state distribution
"""
def generate_river(n=6, small=5, large=10000):
    nA = 2
    nS = n
    p = compute_probabilities(nS, nA)
    r = compute_rewards(nS, nA, small, large)
    mu = compute_mu(nS)
    return nS, nA, p, r, mu


"""
    Compute initial probabilities of the river environment
        @nS: number of states
        @nA: number of actions
        return a probability transition matrix
"""
def compute_probabilities(nS, nA):
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

"""
    Compute reward matrix
        @nS: number of states
        @nA: number of actions
        @small: small reward
        @large: reward
        return the reward matrix
"""
def compute_rewards(nS, nA, small, large):
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
    mu = np.zeros(nS)
    # Starting from the leftmost state
    mu[0] = 1
    return mu
