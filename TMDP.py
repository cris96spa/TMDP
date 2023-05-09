import numpy as np

from discreteEnv import DiscreteEnv

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

    def __init__(self, env:DiscreteEnv,  xi, tau=0, gamma=1):
        self.tau = tau
        self.xi = xi

        # This code works only for an environment that already wrapps discrete environment, otherwise the constructor code won't be resolved correctly
        super(TMDP, self).__init__(gamma)

        P_tau = {s: {a: [] for a in range(self.nA)} for s in range(nS)}
        P_mat_tau = np.zeros(shape=(self.nS * self.nA, self.nS))

        for s in range(self.nS):
            for a in range(self.nA):
                for s1 in range(self.nS):
                    prob = self.P[s][a][s1]
                    prob_tau = prob * (1-tau) + xi[s1]*tau
                    reward = self.reward[s][a][s1]
                    P_tau[s][a].append((prob_tau, s1, reward, False))
                    P_mat_tau[s * self.nA + a][s1] = prob_tau

        self.P_tau = P_tau
        self.P_mat_tau = P_mat_tau