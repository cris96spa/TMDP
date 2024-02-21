"""
    Discrete environment class definition.
"""
import numpy as np

from gym import Env, spaces
from gym.utils import seeding
from model_functions import *


"""
    Discrete Environment class. It presents the following attributes:
        - nS: number of states
        - nA: number of actions
        - P: transitions as a dictionary of dictionary of lists. P[s][a] = [(probability, nextstate, reward, done), ...]
        - mu: initial state distribution as list or array of length nS
        - gamma: discount factor
        - lastaction: used for rendering
        - action_space: action space of the environment
        - observation_space: observation space of the environment

    Args:
        Env (gym.ENV): Environment to be extended to obtain a discrete environment
"""
class DiscreteEnv(Env):

    """
        Constructor
        Args:
            nS (int): number of states
            nA (int): number of actions
            P (dict): Dictionary of dictionary of list. P[s][a] = [(probability, nextstate, reward, done), ...]
            mu (list): initial state distribution [nS]
            gamma (float, optional): discount factor. Default to 1.
            seed (float, optional): pseudo-random generator seed. Default to None.
    """ 
    def __init__(self, nS, nA, P, mu, gamma=1., seed=None) -> None:
        
        #: P (dict): P[s][a] = [(probability, nextstate, reward, done), ...]
        self.P = P

        #: mu (list): initial state distribution [nS]
        self.mu = mu

        #: nS (int): number of states
        self.nS = nS

        #: nA (int): nnumber of actions
        self.nA = nA

        #: gamma (float, optional): discount factor
        self.gamma = gamma

        #: lastaction (): last executed action
        self.lastaction=None

        #: action_space (gym.spaces.discrete.Discrete): discrete action space
        self.action_space = spaces.Discrete(self.nA)

        #: observation_space (gym.spaces.discrete.Discrete): discrete state space
        self.h = spaces.Discrete(self.nS)

        self.seed(seed)
        self.reset()
    
   
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
        
        # Get an initial state, from initial state distribution
        s = categorical_sample(self.mu, self.np_random)
        self.lastaction = None
        # Set the initial state
        self.s = np.array([s]).ravel()
        return self.s
    
    """
        Environment transition step implementation.
        Args:
            -a: the action to be executed
        return:
            next state, the immediate reward, done flag, the probability of that specific transition
    """
    def step(self, a):
        
        # Get the list of possible transitions from the current state, given the action a
        transitions = self.P[self.s[0]][a]
        # Get the probability of moving from s to every possible next state, while picking action a
        probabilities = [t[0] for t in transitions]
        sample = categorical_sample(probabilities, self.np_random)
        p, s, r, done = transitions[sample]
        # update the current state
        self.s = np.array([s]).ravel()
        # update last action
        self.lastaction = a
        
        return self.s, r, {"done":done}, p
