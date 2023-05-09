"""
    Discrete environment class definition.
"""
import numpy as np

from gym import Env, spaces
from gym.utils import seeding

"""
    Sample from categorical distribution
        @prob_n : probability distribution vector
        @np_random: random number generator
        return: a categorical state drown from prob_n
"""
def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    # Compute cumulative sum of the probability vector
    csprob_n = np.cumsum(prob_n)
    print(csprob_n)
    return (csprob_n > np_random.rand()).argmax()

"""
    Discrete Environment class. It presents the following attributes:
    - nS: number of states
    - nA: number of actions
    - P: transitions as a dictionary of dictionary of lists. P[s][a] = [(probability, nextstate, reward, done), ...]
    - isd: initial state distribution as list or array of length nS
    - gamma: discount factor
    - lastaction: used for rendering
    - action_space: action space of the environment
    - observation_space: observation space of the environment
"""
class DiscreteEnv(Env):
    def __init__(self, nS, nA, P, isd, gamma=1) -> None:
        self.P = P
        self.isd = isd
        self.nS = nS
        self.nA = nA
        self.gamma = gamma

        self.lastaction=None

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self._reset()
    
    """
        Set a seed for reproducibility of results
    """
    def seed(self, seed=None):
        # set a random generator
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """
        Reset the environment to an initial state
    """
    def reset(self):
        # Get an initial state, from inistial state distribution
        s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        # Set the initial state
        self.s = np.array([s]).ravel()
        return self.s
    
    """
        Environment transition step implementation
            @a: the action to be executed
            return next state, the immediate reward, a done flag and the probability of that specific transition
    """
    def step(self, a):
        """
            Get the probability transition associated to state s and action a. 
            P[s][a][0] is a vector of length nS, telling the probability of moving from state s, to each other state s' in S, picking action a
            
        """
        transitions = self.P[np.asscalar(self.s)][np.asscalar(a)]
        sample = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[sample]
        
        # update the current state
        self.s = np.array([s]).ravel()
        # update last action
        self.lastaction = a
        return self.s, r, d, {"prob":p}
