import numpy as np
from model_functions import *
import pygame 
from RiverSwim import RiverSwim
from gymnasium.envs.toy_text.utils import categorical_sample
from typing import List, Optional
from gymnasium import Env, spaces, utils, logger

"""
    A Teleport-MDP (TMDP) is a Markovian decision process that follows (1 - tau) times the model dynamics,
    while tau times the state teleport probability distribution xi
    It presents the following attributes:
        - tau (float, optional): teleport probability. Default to 0.
        - xi (numpy.ndarray): state teleport probability distribution
        - reward (numpy.ndarray): rewards associated to each action for each state [ns, nA, nS]
        - P_mat (numpy.ndarray): Matrix probability of moving from state s to s' (for each pairs (s,s') when picking action a (for each a) [nS, nA, nS]
        - allowed_actions (list): List of allowed action for each state
        - P_tau (dict): Dictionary of dictionary of list. P_tau[s][a] = [(probability, nextstate, reward, done), ...]
        - P_mat_tau (numpy.ndarray): Matrix probability of moving from state s to s' considering the probability of teleporting (for each pairs (s,s') when picking action a (for each a) [nS, nA, nS]

    Args:
        DiscreteEnv (gym.ENV): Implementation of a discrete environment, from the gym.ENV class.

"""
metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

class TMDP(Env):
    """
        Constructor

        Args:
            env (DiscreteEnv): Environmental class to be extended to obtain a TMDP
            xi (numpy.ndarray): state teleport probability distribution
            tau (float, optional): teleport probability. Default to 0.
            gamma: discount factor. Default to 0.99
    """
    def __init__(self, env:Env,  
                 xi, tau=0., gamma=0.99, 
                 seed=None, discount_tau:bool=True,
                 render_mode: Optional[str] = None,
                 xi_schedule=None):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        #: xi (numpy.ndarray): state teleport probability distribution
        self.xi = xi
        self.gamma = gamma
        self.nS = env.nS
        self.nA = env.nA
        self.discount_tau = discount_tau
        self.xi_schedule = xi_schedule
        # Set the value of tau and build the P_tau and P_mat_tau
        #self.seed(seed)
        self.update_tau(tau)
        self.reset()

        self.render_mode = self.env.render_mode = render_mode

    """
        Basic step implementation. Allow to perform a single step in the environment.

        Args:
            a (int): action to be taken

        Returns:
            (int, float, bool, float): next state, immmediate reward, done flag, probability of ending up in that state
    """
    def step(self, a):
        if self.env.np_random.random() <= self.tau:
            # Teleport branch
            s_prime = categorical_sample(self.xi, self.env.np_random)
            self.env.lastaction = a
            #r = self.env.reward[int(self.env.s), a, int(s_prime)]
            r = 0
            if self.discount_tau:
                r = r * self.tau
            self.env.s = s_prime
            done = self.env.is_terminal(self.env.s)

            if self.env.render_mode == "human":
                self.env.render()

            prob = self.xi[s_prime]#*self.tau
            # In this case the done flag signal that a teleport happened
            
            return self.env.s, r, {"done":done, "teleport": True}, {"prob":prob}
        else:
            #print("Following regular probability transition function")
            s_prime, reward, flags, prob = self.env.step(a)
            flags["teleport"] = False
            if self.discount_tau:
                reward = reward * (1 - self.tau)

            if self.render_mode == "human":
                self.render()

            return s_prime, reward, flags, prob
    
    def render(self):
        self.env.render()

    """
        Update the teleport probability tau
        Args:
            tau (float): new teleport probability
    """
    def update_tau(self, tau):
        self.tau = tau
        if self.xi_schedule is not None:
            self.xi = self.xi_schedule(tau, self.nS)

    def compute_teleport_matrix(self):
        if self.tau == 0:
            P_mat_tau = self.env.P_mat
        else:
            P_mat_tau = (1 - self.tau) * self.env.P_mat #+ self.tau * self.xi[None, None, :]  # Broadcasting xi
        return P_mat_tau

    def seed(self, seed=None):
        # set a random generator
        #seed = self.env.seed(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return self.env.reset(seed=seed)
