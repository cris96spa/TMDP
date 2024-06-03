"""
    Discrete environment class definition.
"""
import numpy as np

from gymnasium import Env, spaces
from gymnasium.utils import seeding
from model_functions import *


"""
    Discrete Environment class.
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

    def __init__(self, nS, nA, P, mu, gamma=1., seed=None, render_mode=None) -> None:
        pass

    def is_terminal(self, state):
        
        pass
