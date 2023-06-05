import numpy as np

from DiscreteEnv import DiscreteEnv
from model_functions import *

"""
    A Teleport-MDP is a Markovia decision process that follows (1 - tau) times the model dynamics,
    while tau times following the state teleport probability distribution xi
    - nS: number of states
    - nA: number of actions
    - P: transitions as a dictionary of dictionary of lists. P[s][a] = [(probability, nextstate, reward, done), ...]
    - isd: initial state distribution as list or array of length nS
    - gamma: discount factor
    - tau: teleport probability
    - xi: state teleport probability distribution
"""
class TMDP(DiscreteEnv):

    def __init__(self, env:DiscreteEnv,  xi, tau=0, gamma=1, seed=None):
        self.tau = tau
        self.xi = xi
        self.reward = env.reward
        self.P_mat = env.P_mat
        self.allowed_actions = env.allowed_actions

        # This code works only for an environment that already wrapps discrete environment, otherwise the constructor code won't be resolved correctly
        super(TMDP, self).__init__(env.nS, env.nA, env.P, env.mu, gamma, seed)
        
        if tau == 0:
            P_tau = self.P
            P_mat_tau = self.P_mat
        else:
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

    # Sistemare impementazione
    def step(self, a, seed=None, debug=False):
        np.random.seed(seed)
        if not debug:
            if np.random.rand() <= self.tau:
                # Teleport branch
                states = [i for i in range(self.nS)]
                s_prime = np.random.choice(states, p=self.xi)
                #print("Teleported from state {} to {}:".format(self.s, s_prime))
                self.lastaction = a
                r = self.reward[self.s, a, s_prime]
                self.s = np.array([s_prime]).ravel()
                return self.s, r, True, {"prob:", self.xi[s_prime]}
            else:
                #print("Following regular probability transition function")
                return super(TMDP, self).step(a)
        else:
            transitions = self.P_tau[self.s[0]][a]
            sample = categorical_sample([t[0] for t in transitions], self.np_random)
            p, s, r, d = transitions[sample]
            # update the current state
            self.s = np.array([s]).ravel()
            # update last action
            self.lastaction = a
            return self.s, r, d, {"prob":p}


# TBD tenere traccia del teleporting 