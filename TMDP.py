import numpy as np
from DiscreteEnv import DiscreteEnv
from model_functions import *

"""
    A Teleport-MDP (TMDP) is a Markovian decision process that follows (1 - tau) times the model dynamics,
    while tau times the state teleport probability distribution xi

    Args:
        DiscreteEnv (gym.ENV): Implementation of a discrete environment, from the gym.ENV class.
"""
class TMDP(DiscreteEnv):
    """
        Constructor

        Args:
            env (DiscreteEnv): Environmental class to be extended to obtain a TMDP
            xi (numpy.ndarray): state teleport probability distribution
            tau (float, optional): teleport probability. Default to 0.
            gamma: discount factor. Default to 0.99
            seed (float, optional): pseudo-random generator seed. Default to None.
    """
    def __init__(self, env:DiscreteEnv,  xi, tau=0., gamma=0.99, seed=None):
        
        #: tau (float, optional): teleport probability
        self.tau = tau
        
        #: xi (numpy.ndarray): state teleport probability distribution
        self.xi = xi
        
        #: reward (numpy.ndarray): rewards associated to each action for each state []
        self.reward = env.reward
        
        #: P_mat (numpy.ndarray): Matrix probability of moving from state s to s' (for each pairs (s,s') when picking action a (for each a) [nS*nA, nS]
        self.P_mat = env.P_mat
        #: allowed_actions (list): List of allowed action for the defined problem  
        self.allowed_actions = env.allowed_actions


        # This code works only for an environment that already wrapps discrete environment, otherwise the constructor code won't be resolved correctly
        super(TMDP, self).__init__(env.nS, env.nA, env.P, env.mu, gamma, seed)

        if tau == 0:
            # Original problem
            P_tau = self.P
            P_mat_tau = self.P_mat
        else:
            # Simplified problem
            P_tau = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
            P_mat_tau = np.zeros(shape=(self.nS * self.nA, self.nS))

            for s in range(self.nS):
                for a in range(self.nA):
                    for s1 in range(self.nS):
                        prob = self.P[s][a][s1][0]
                        prob_tau = prob * (1-tau) + xi[s1]*tau
                        reward = self.reward[s][a][s1]
                        P_tau[s][a].append((prob_tau, s1, reward, False))
                        P_mat_tau[s * self.nA + a][s1] = prob_tau

        self.P_tau = P_tau
        self.P_mat_tau = P_mat_tau


    def step(self, a, seed=None, debug=False):
        """
        Basic step implementation. Allow to perform a single step in the environment.

        Args:
            a (int): action to be taken
            seed (float, optional): pseudo-random generator seed. Default to None.
            debug (bool, optional): Used for debug purposes. Defaults to False.

        Returns:
            (int, float, bool, float): next state, immmediate reward, done flag, probability of ending up in that state
        """
        np.random.seed(seed)
        if not debug:

            if np.random.rand() <= self.tau:
                # Teleport branch
                # If a teleport occurred, we can actually stop learning
                states = [i for i in range(self.nS)]
                s_prime = np.random.choice(states, p=self.xi)
                #print("Teleported from state {} to {}:".format(self.s, s_prime))
                self.lastaction = a
                r = self.reward[self.s, a, s_prime]
                self.s = np.array([s_prime]).ravel()
                # In this case the done flag signal that a teleport happened
                return self.s, r, True, self.xi[s_prime]
            else:
                #print("Following regular probability transition function")
                return super(TMDP, self).step(a)
        """ else:
            transitions = self.P_tau[self.s[0]][a]
            sample = categorical_sample([t[0] for t in transitions], self.np_random)
            p, s, r, d = transitions[sample]
            # update the current state
            self.s = np.array([s]).ravel()
            # update last action
            self.lastaction = a
            return self.s, r, d, {"prob":p}"""


# TBD tenere traccia del teleporting 