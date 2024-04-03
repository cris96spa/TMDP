import numpy as np
from DiscreteEnv import DiscreteEnv
from model_functions import *
import pygame 
from River_swim import River
"""
    A Teleport-MDP (TMDP) is a Markovian decision process that follows (1 - tau) times the model dynamics,
    while tau times the state teleport probability distribution xi
    It presents the following attributes:
        - tau (float, optional): teleport probability. Default to 0.
        - xi (numpy.ndarray): state teleport probability distribution
        - reward (numpy.ndarray): rewards associated to each action for each state [ns, nA, size]
        - P_mat (numpy.ndarray): Matrix probability of moving from state s to s' (for each pairs (s,s') when picking action a (for each a) [size, nA, size]
        - allowed_actions (list): List of allowed action for each state
        - P_tau (dict): Dictionary of dictionary of list. P_tau[s][a] = [(probability, nextstate, reward, done), ...]
        - P_mat_tau (numpy.ndarray): Matrix probability of moving from state s to s' considering the probability of teleporting (for each pairs (s,s') when picking action a (for each a) [size, nA, size]

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
        # Set the value of tau and build the P_tau and P_mat_tau
        self.update_tau(tau)


    """
        Basic step implementation. Allow to perform a single step in the environment.

        Args:
            a (int): action to be taken

        Returns:
            (int, float, bool, float): next state, immmediate reward, done flag, probability of ending up in that state
    """
    def step(self, a):
        s = self.env.state
        if self.env.np_random.random() <= self.tau:
            # Teleport branch
            s_prime = self.env.np_random.choice(len(self.xi), p=self.xi)
            #print("Teleported from state {} to {}:".format(self.state, s_prime))
            self.lastaction = a
            #r = self.env.reward[self.state, a, s_prime]
            r = 0.
            self.env.state = np.array([s_prime]).ravel()
            prob = self.xi[s_prime]*self.tau
            # In this case the done flag signal that a teleport happened
            return self.env.state, r, {"done":self.env.is_terminal(self.env.state[0]), "teleport": True}, {"prob":prob}
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
        if isinstance(self.env, River):
            if tau == 0:
                # Original problem
                P_tau = self.env.P
                P_mat_tau = self.env.P_mat
            else:
                # Simplified problem
                P_tau = {s: {a: [] for a in range(self.env.nA)} for s in range(self.env.size)}
                P_mat_tau = np.zeros(shape=(self.env.size, self.env.nA, self.env.size))

                for s in range(self.env.size):
                    for a in range(self.env.nA):
                        for s1 in range(self.env.size):
                            prob = self.env.P[s][a][s1][0]
                            prob_tau = prob * (1-tau) + self.xi[s1]*tau
                            reward = self.env.reward[s][a][s1]
                            P_tau[s][a].append((prob_tau, s1, reward, self.env.is_terminal(s1)))
                            P_mat_tau[s][a][s1] = prob_tau

            self.P_tau = P_tau
            self.P_mat_tau = P_mat_tau

    def reset(self):
        return self.env.reset()