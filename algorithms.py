import numpy as np
from gym import Env
from distanceMeasure import *
from scipy.special import softmax
from model_functions import *


"""
    Compute Q* using bellman optimality operator iteratively
    @nS: number of states
    @nA: number of actions
    @P_mat: probability transition function
    @reward: reward function
    @epsilon: stopping threshold value
    @gamma: discount factor
    return: Q* estimated as an iterative application of the bellman optimality operator until |T*(Q)-Q|<=epsilon
"""
def bellman_optimal_q(nS, nA, P_mat, reward, epsilon, gamma):
    r_s_a = compute_r_s_a(nS, nA, P_mat, reward)
    Q = np.zeros((nS, nA))
    loop = True
    while loop:
        Q_old = Q.copy()
        for s in range(nS):
            for a in range(nA):
                sum = 0
                a_star = 0
                for s_prime in range(nS):
                    a_star = np.argmax(Q[s_prime])
                    sum = sum + P_mat[s*nA + a][s_prime]*Q[s_prime, a_star]
                Q[s,a] = r_s_a[s, a] + gamma*sum
        delta_q = np.linalg.norm(Q - Q_old, np.inf)
        if delta_q <= epsilon:
            loop = False
    return Q

"""
    Epsilon greedy action selection
        @s: current state
        @Q: current state action value function
        @eps: exploration/exploration factor
        @allowed_actions: actions allowed in the given state s
        return an epsilon greedy action choice
"""
def eps_greedy(s, Q, eps, allowed_actions):
    # epsilon times pick an action uniformly at random (exploration)
    if np.random.rand() <= eps:
        actions = np.where(allowed_actions)
        # Extract indices of allowed actions
        actions = actions[0]
        # pick a uniformly random action
        a = np.random.choice(actions, p=(np.ones(len(actions))/len(actions)))
        #print("Random action picked: ",a)
    else:
        # Extract the Q function for the given state
        Q_s = Q[s, :].copy()
        
        # Set to -inf the state action value function of not allowed actions
        Q_s[allowed_actions == 0] = -np.inf
        # Pick the most promizing action (exploitation)
        a = np.argmax(Q_s)
        #print("Greedy action picked: ",a)
    return a

"""
    SARSA algorithm implementation
        @env: environment object
        @s: current state
        @a: first action to be taken
        @Q: current state-action value function
        @M: number of iterations to be considered
        return the state action value function under the pseudo-optimal policy found
"""
def SARSA(env:Env, s, a, Q, M=5000):
    m = 1
    # SARSA main loop
    while m < M:
        # Learning rate initialization
        alpha = (1- m/M)
        # epsilon update
        eps = (1 - m/M)**2

        # Perform a step in the environment, picking action a
        s_prime, r, d, p = env.step(a)

        # Policy improvement step
        # N.B. allowed action is not present in the Env object, must be managed
        a_prime = eps_greedy(s_prime, Q, eps, env.allowed_actions[s_prime.item()])
        # Evaluation step
        Q[s,a] = Q[s,a] + alpha*(r + env.gamma*Q[s_prime, a_prime] - Q[s,a])
        # Setting next iteration
        m = m+1
        s = s_prime
        a = a_prime
    return Q

"""
    Q_learning algorithm implementation
        @env: environment object
        @s: current state
        @a: first action to be taken
        @Q: current state-action value function
        @M: number of iterations to be considered
        @status: intermediate results flag
        return the state action value function under the pseudo-optimal policy found
"""
def Q_learning(env:Env, s, a, Q, M=5000, status_step=200):
    m = 1
    # SARSA main loop
    J = []
    while m < M:
        # Learning rate initialization
        alpha = (1- m/M)
        # epsilon update
        eps = (1 - m/M)**2
        # Perform a step in the environment, picking action a
        s_prime, r, d, p = env.step(a)

        # Policy improvement step
        # N.B. allowed action is not present in the Env object, must be managed
        a_prime = eps_greedy(s_prime, Q, eps, env.allowed_actions[s_prime.item()])

        #print("Step:", m, " state:", s, " action:", a, " next state:",s_prime, " reward:",r, " next action:", a_prime)
        # Evaluation step
        Q[s,a] = Q[s,a] + alpha*(r + env.gamma*np.max(Q[s_prime, :]) - Q[s,a])
        if(m % status_step == 0):
            J.append(get_expected_avg_reward(env.P_mat, get_policy(Q), env.reward, env.gamma, env.mu))
        # Setting next iteration
        m = m+1
        s = s_prime
        a = a_prime
    return Q, J

"""
    Compare two different policies in terms of a distance measure
        @measure: the measure to be used for the comparison
        @pi: a policy row vector
        @pi_prime: a second policy row vector
        return the difference, according to the given measure, of the two policies
"""
def compare_policies(measure:DistanceMeasure, pi, pi_prime):
    return measure.compute_distance(pi, pi_prime)

