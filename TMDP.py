import numpy as np
from DiscreteEnv import DiscreteEnv
from model_functions import *
import pygame 
from River_swim import River
from gymnasium.envs.toy_text.utils import categorical_sample
from typing import List, Optional

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
class TMDP(Env):
    """
        Constructor

        Args:
            env (DiscreteEnv): Environmental class to be extended to obtain a TMDP
            xi (numpy.ndarray): state teleport probability distribution
            tau (float, optional): teleport probability. Default to 0.
            gamma: discount factor. Default to 0.99
    """
    def __init__(self, env:Env,  xi, tau=0., gamma=0.99, seed=None):
        self.env = env
        #: xi (numpy.ndarray): state teleport probability distribution
        self.xi = xi
        self.gamma = gamma
        self.nS = env.nS
        self.nA = env.nA
        # Set the value of tau and build the P_tau and P_mat_tau
        self.update_tau(tau)
        self.reset()

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
            r = self.env.reward[int(self.env.s), a, int(s_prime)]
            self.env.lastreward = r
            self.env.s = s_prime
            done = self.env.is_terminal(self.env.s)
            if done:
                s_prime, _ = self.env.reset()

            if self.env.render_mode == "human":
                self.env.render()

            prob = self.xi[s_prime]*self.tau
            # In this case the done flag signal that a teleport happened
            
            return self.env.s, r, {"done":done, "teleport": True}, {"prob":prob}
        else:
            #print("Following regular probability transition function")
            s_prime, reward, flags, prob = self.env.step(a)
            prob["prob"] = prob["prob"]*(1-self.tau)
            flags["teleport"] = False
            return s_prime, reward, flags, prob

    """
        Update the teleport probability tau, and the associated transition probabilities P_tau and P_mat_tau
        Args:
            tau (float): new teleport probability
    """
    def update_tau(self, tau):
        self.tau = tau
        if tau == 0:
            P_mat_tau = self.env.P_mat
        else:
            # Simplified problem
            P_mat_tau = (1 - tau) * self.env.P_mat + tau * self.xi[None, None, :]  # Broadcasting xi

        self.P_mat_tau = P_mat_tau

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return self.env.reset(seed=seed)