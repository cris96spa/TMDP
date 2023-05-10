import numpy as np

from discreteEnv import DiscreteEnv
from river_swim_generator import generate_river

"""
    A river swim environment is an environment in which there are several sequential states and
    only two possible moves, left and right. It is assumed that left move is in the same direction of the river
    flow, hence it always lead to the immediatly left state, whereas the right move is done against the flow, 
    meaning that it has a very small probability of leading to the left, a pretty high probability of remaining 
    in the same state and a small probability of moving to the right. The right you are able to move, the higher
    will be rewards.
"""

class River(DiscreteEnv):

    def __init__(self, gamma=1):

        # Generate river parameters using the auxiliary function    
        nS, nA, p, r, mu = generate_river()

        # Parameter initialization
        self.reward = r
        # Creating the dictionary of dictionary of lists that represents P
        P = {s: {a :[] for a in range(nA)} for s in range(nS)}
        # Probability matrix of the problem dynamics
        P_mat = np.zeros(shape=(nS*nA, nS))
        self.allowed_actions = []

        # Assigning values to P and P_mat
        for s in range(nS):
            self.allowed_actions.append([1,1])
            for a in range(nA):
                for s1 in range(nS):
                    # Get the probability of moving from s->s1, when action a is picked
                    prob = p[s][a][s1]
                    # Get the reward associated to the transition from s->s1, when a is picked
                    reward = r[s][a][s1]
    
                    # Build P[s][a] that is a list of tuples, containint the probability of that move, the next state, the associated reward and a done flag
                    P[s][a].append((prob, s1, reward, False))

                    # Assign P_mat values
                    P_mat[s*nA + a][s1] = prob
                    
        self.P_mat = P_mat
        # Calling the superclass constructor to initialize other parameters
        super(River, self).__init__(nS, nA, P, mu, gamma)