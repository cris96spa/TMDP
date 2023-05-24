import numpy as np
from gym import Env
from distanceMeasure import *
from scipy.special import softmax
np.set_printoptions(precision=3)

"""
    Compute the average reward when picking action a in state s
    @nS: number of states
    @nA: number of actions
    @P_mat: probability transition function
    @reward: the reward function
    return:the average reward when picking action a in state s as a |S| x |A| matrix
"""
def compute_r_s_a(nS, nA, P_mat, reward):
    # Average reward when taking action a in state s, of size |S|x|A|
    r_s_a =np.zeros(shape=(nS, nA))
    for s in range(nS):
        for a in range(nA):
            avg_rew = 0
            for s_prime in range(nS):
                avg_rew = avg_rew + P_mat[s*nA + a][s_prime] * reward[s, a, s_prime]
            #print("Updating state {}, action {}, with {}".format(s, a, avg_rew))
            r_s_a[s, a] = avg_rew
    return r_s_a

"""
    Compute the probability of moving from state s to state sprime, under policy pi
        @P_mat: probability transition function
        @pi: the given policy
        return: the probability of moving from state s to state sprime under policy pi, as a [nS x nS] matrix 
"""
def compute_p_sprime_s(P_mat, pi):
    nS = pi.shape[0]
    nA = pi.shape[1]
    P_sprime_s = np.zeros((nS, nS))
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                P_sprime_s[s][s_prime] = P_sprime_s[s][s_prime] + pi[s, a] * P_mat[s*nA + a][s_prime]
    return P_sprime_s


"""
    Compute the discounted state distribution
        @mu: initial state distribution
        @P_mat: probability transition function
        @pi: the given policy
        @gamma: discount factor
        return: the discount state distribution as a vector of |S| elements
"""
def compute_d(mu, P_mat, pi, gamma):
    nS = pi.shape[0]
    nA = pi.shape[1]
    I = np.eye(nS)
    P_sprime_s = compute_p_sprime_s(P_mat, pi)
    inv = np.linalg.inv(I - gamma*P_sprime_s)
    d = np.matmul((1-gamma)*mu, inv)
    return d


"""
    Extract the policy from a given state action value function
        @Q: the state action value function
        @det: deterministic flag. Whether or not extracting a deterministic policy
        return the greedy policy according to Q, as an |S|x|A| matrix
"""
def get_policy(Q, det=True):
    pi = np.zeros(Q.shape)
    if det:
        for x in range(Q.shape[0]):
            pi[x,np.argmax(Q[x])] = 1
    else:
        for x in range(Q.shape[0]):
           pi[x] = softmax(Q[x]) 

    return pi


"""
    Extract the value function from a given state action value function
        @Q: the state action value function
        return the value function as V(s) = max{a in A}{Q(s,a)}
"""
def get_value_function(Q, det=True):
    pi = get_policy(Q, det)
    V = np.zeros(Q.shape[0])
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            V[s] = V[s] + Q[s,a]*pi[s,a]
    return V


"""
    Compute the expected discounted sum of returns
        @r_s_a: the average reward when picking action a in state s as an |S| x |A| matrix
        @pi: the given policy
        @gamma: discount factor
        return: the expected discounted sum of returns as a scalar value
"""
def compute_j(r_s_a, pi, d, gamma):
    nS = pi.shape[0]
    nA = pi.shape[1]
    J = 0
    for s in range(nS):
        for a in range(nA):
            J = J + pi[s,a] * r_s_a[s, a]*d[s]
    J = J/(1-gamma)
    return J

"""
    Utility function to get the expected discounted sum of returns
        @P_mat: probability transition function
        @pi: the given policy
        @reward: the reward function
        @gamma: discount factor
        return: the expected discounted sum of returns as a scalar value
"""
def get_expected_avg_reward(P_mat, pi, reward, gamma, mu):
    nS = pi.shape[0]
    nA = pi.shape[1]
    r_s_a = compute_r_s_a(nS, nA, P_mat, reward)
    d = compute_d(mu, P_mat, pi, gamma)
    return compute_j(r_s_a, pi, d, gamma)

"""
    Compute the state action nextstate value function
        @nS: number of states
        @nA: number of actions
        @r_s_a: the average reward when picking action a in state s as an |S| x |A| matrix
        @gamma: discount factor
        @V: the state value function
        return: the state action nextstate value function as an |S|x|A|x|S| matrix
"""
def compute_state_action_nextstate_value_function(nS, nA, r_s_a, gamma, V):
    U = np.zeros((nS, nA, nS))
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                U[s, a, s_prime] = r_s_a[s, a] + gamma*V[s_prime]
    return U

"""
    Utility function to get the state action nextstate value function
        @P_mat: probability transition function
        @reward: the reward function
        @gamma: discount factor
        @Q: the state action value function
        return: the state action nextstate value function as an |S|x|A|x|S| matrix
"""
def get_state_action_nextstate_value_function(P_mat, reward, gamma, Q, det=True):
    nS = Q.shape[0]
    nA = Q.shape[1]
    r_s_a = compute_r_s_a(nS, nA, P_mat, reward)
    V = get_value_function(Q, det)
    return compute_state_action_nextstate_value_function(nS, nA, r_s_a, gamma, V)


"""
    Compute the state policy advantage function
        @nS: number of states
        @nA: number of actions
        @Q: state action value function
        @V: state value function
        return: the policy advantage function as an |S|x|A| matrix
"""
def compute_policy_advantage_function(nS, nA, Q, V):
    A = np.zeros(Q.shape)
    for s in range(nS):
        for a in range(nA):
            A[s, a] = Q[s,a] - V[s]
    return A

"""
    Utility function to get policy advantage function
        @Q: the state action value function
        @det: deterministic flag. Whether or not extracting a deterministic policy
        return: the state policy advantage function as an |S|x|A| matrix
"""
def get_policy_advantage_function(Q, det):
    nS, nA = Q.shape
    V = get_value_function(Q, det)
    return compute_policy_advantage_function(nS, nA, Q, V)

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
        return the state action value function under the pseudo-optimal policy found
"""
def Q_learning(env:Env, s, a, Q, M=5000):
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

        #print("Step:", m, " state:", s, " action:", a, " next state:",s_prime, " reward:",r, " next action:", a_prime)
        # Evaluation step
        Q[s,a] = Q[s,a] + alpha*(r + env.gamma*np.max(Q[s_prime, :]) - Q[s,a])
        # Setting next iteration
        m = m+1
        s = s_prime
        a = a_prime
    return Q

"""
    Compare two different policies in terms of a distance measure
        @measure: the measure to be used for the comparison
        @pi: a policy row vector
        @pi_prime: a second policy row vector
        return the difference, according to the given measure, of the two policies
"""
def compare_policies(measure:DistanceMeasure, pi, pi_prime):
    return measure.compute_distance(pi, pi_prime)

