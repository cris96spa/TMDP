import numpy as np

from DiscreteEnv import DiscreteEnv
from river_swim_generator import generate_river

"""
    A river swim environment is an environment in which there are several sequential states and
    only two possible moves, left and right. It is assumed that left move is in the same direction of the river
    flow, hence it always lead to the immediatly left state, whereas the right move is done against the flow, 
    meaning that it has a ver'y small probability of leading to the left, a pretty high probability of remain'ing 
    in the same state and a small probability of moving to the right. The right you are able to move, the higher
    will be rewards.
    
    It presents the following attributes:
        - reward (np.ndarray): rewards associated to each action for each state [ns, nA, nS]
        - P_mat (np.ndarray): Matrix probability of moving from state s to s' (for each pairs (s,s') when picking action a (for each a) [nS, nA, nS]
        - allowed_actions (list): List of allowed action for each state

    Args:
        DiscreteEnv (gym.ENV): Implementation of a discrete environment, from the gym.ENV class.
"""
class River(DiscreteEnv):
    
    """Constructor

        Args:
            nS (int): _description_
            gamma (float, optional): discount factor. Default to 1.
            small (int, optional): small reward. Defaults to 5.
            large (int, optional): large reward. Defaults to 10000.
            seed (float, optional): pseudo-random generator seed. Default to None.
    """

    def __init__(self, nS, gamma=1., small=5, large=10000, seed=None):
        
        # Generate river parameters using the auxiliary function    
        nS, nA, p, r, mu = generate_river(nS, small, large)

        # Parameter initialization
        self.reward = r
        # Creating the dictionary of dictionary of lists that represents P
        P = {s: {a :[] for a in range(nA)} for s in range(nS)}
        # Probability matrix of the problem dynamics
        P_mat = np.zeros(shape=(nS, nA, nS))
        self.allowed_actions = []

        # Assigning values to P and P_mat
        for s in range(nS):
            # Add allowed actions (left, right) for each state
            self.allowed_actions.append([1,1])

            for a in range(nA):
                for s1 in range(nS):
                    # Get the probability of moving from s->s1, when action a is picked
                    prob = p[s][a][s1]
                    # Get the reward associated to the transition from s->s1, when a is picked
                    reward = r[s][a][s1]
    
                    # Build P[s][a] that is a list of tuples, containint the probability of that move, the next state, the associated reward and a termination flag
                    # The termination flag is set to True if the reward is different from 0, meaning that the agent reached the goal
                    P[s][a].append((prob, s1, reward, reward !=0))

                    # Assign P_mat values
                    P_mat[s][a][s1] = prob
                    
        self.P_mat = P_mat
        # Calling the superclass constructor to initialize other parameters
        super(River, self).__init__(nS, nA, P, mu, gamma, seed)


        