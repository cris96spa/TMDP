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
            self.lastaction = a
            r = self.env.reward[int(self.env.s), a, int(s_prime)]
            self.env.s = s_prime

            if self.env.render_mode == "human":
                self.env.render()

            prob = self.xi[s_prime]*self.tau
            # In this case the done flag signal that a teleport happened
            done = self.env.is_terminal(self.env.s) and r != 0
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
            # Original problem
            P_tau = self.env.P
            P_mat_tau = self.env.P_mat
        else:
            # Simplified problem
            P_tau = {s: {a: [] for a in range(self.env.nA)} for s in range(self.env.nS)}
            P_mat_tau = np.zeros(shape=(self.env.nS, self.env.nA, self.env.nS))

            for s in range(self.env.nS):
                for a in range(self.env.nA):
                    for s1 in range(self.env.nS):
                        prob = self.env.P_mat[s][a][s1]
                        prob_tau = prob * (1-tau) + self.xi[s1]*tau
                        reward = self.env.reward[s][a][s1]
                        P_tau[s][a].append((prob_tau, s1, reward, self.env.is_terminal(s1)))
                        P_mat_tau[s][a][s1] = prob_tau

        self.P_tau = P_tau
        self.P_mat_tau = P_mat_tau

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return self.env.reset(seed=seed)