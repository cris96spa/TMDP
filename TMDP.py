import numpy as np

from .discreteEnv import DiscreteEnv

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

    def __init__(self, nS, nA, P, isd, gamma, tau, xi) -> None:
        super().__init__(nS, nA, P, isd, gamma)
        self.tau = tau
        self.xi = xi

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        P_mat = np.zeros(shape=(nS * nA, nS))
        P_mat_tau = np.zeros(shape=(nS * nA, nS))

        for s in range(nS):
            for a in range(nA):
                for s1 in range(nS):
                    prob = p[s][a][s1]
                    reward = r[s][a][s]
                    P[s][a].append((prob, s1, reward, False))
                    P_mat[s * nA + a][s1] = prob
                    P_mat_tau[s * nA + a][s1] = prob * (1-tau) + xi[s1]*tau

        self.P_mat = P_mat
        self.P_mat_tau = P_mat_tau