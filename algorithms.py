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
def Q_learning(env:Env, s, a, Q, Q_star, M=5000, alpha=0., status_step=200, debug=False, main_p=True):
    m = 0
    J_main_p = []
    J_curr_p = []
    delta_q = []
    delta_J = []
    l_bounds = []
    nS, nA = Q.shape
    visits = np.zeros(nS)
    dec_alpha = np.ones(nS)*alpha
    # Q_learning main loop
    while m < M:
        # Learning rate initialization
        visits[s] = visits[s]+1
        # epsilon update
        eps = (1 - m/M)**2
        # Perform a step in the environment, picking action a
        s_prime, r, d, p = env.step(a, debug=debug)

        # Policy improvement step
        # N.B. allowed action is not present in the Env object, must be managed
        a_prime = eps_greedy(s_prime, Q, eps, env.allowed_actions[s_prime.item()])

        #print("Step:", m, " state:", s, " action:", a, " next state:",s_prime, " reward:",r, " next action:", a_prime)
        # Evaluation step
        Q[s,a] = Q[s,a] + dec_alpha[s]*(r + env.gamma*np.max(Q[s_prime, :]) - Q[s,a])
        if(m % status_step == 0):
            # Compute performance on the original problem
            J_0 = get_expected_avg_reward(env.P_mat, get_policy(Q), env.reward, env.gamma, env.mu)
            J_main_p.append(J_0)

            # Compute performance on the current problem
            J_p = get_expected_avg_reward(env.P_mat_tau, get_policy(Q), env.reward, env.gamma, env.mu)
            J_curr_p.append(J_p)

            ### Compute the lower bound on performance improvement ###
            # Compute the discount state distribution
            d = compute_d(env.mu, env.P_mat_tau, get_policy(Q), env.gamma)
            # Compute the gamma discounted state distribution
            delta = compute_delta(d, get_policy(Q))
            # Compute the state value function
            V = get_value_function(Q) 
            # Compute the expected reward when picking action a in state s
            r_s_a = compute_r_s_a(nS, nA, env.P_mat_tau, env.reward)
            # Compute the state action next-state value function U_tau(s,a,s') = R(s,a) + \gamma*V_tau(s')
            U = compute_state_action_nextstate_value_function(nS, nA, r_s_a, env.gamma, V)
            # Rebuild Q using U as Q_tau(s,a) = \sum{s' \in S}P_tau(s'|s,a)*U_tau(s,a,s')
            Q_t = rebuild_Q_from_U(env.P_mat_tau, U)

            # Compute the relative model advantage function hat \hat{A}_{tau, mu}(s,a)
            A_tau_hat = compute_relative_model_advantage_function_hat(env.P_mat, env.xi, U)
            # Compute the discounted distribution relative model advantage function hat \hat{A}_{tau, mu}
            A_hat = compute_discounted_distribution_relative_model_advantage_function_hat(A_tau_hat, delta)
            # The dissimilarity term D = D_e * gamma * D_inf is upperbounded by 4*gamma+(tau - tau_1)
            # Compute Delta Q_tau as the superior among the difference of the L_1 norm of elements of Q_tau
            d_q_t = get_sup_difference_q(Q_t)
            
            # Compute the performance improvement lower bound when moving to tau=0
            l_b = compute_performance_improvement_lower_bound(A_hat, env.gamma, d_q_t, env.tau, 0)

            l_bounds.append(l_b)

            # Compute the empirical performance improvement when moving to tau=0
            delta_J.append(J_0 - J_p )

            delta_q.append(np.linalg.norm(Q - Q_star, np.inf))

        # Setting next iteration
        m = m+1
        s = s_prime
        a = a_prime
        dec_alpha[s] = max(0, alpha*(1- visits[s]*nS/M))
    return Q, J_main_p, J_curr_p, delta_q, delta_J, l_bounds

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
def batch_Q_learning(env:Env, s, a, Q, Q_star, batch_size, M=5000, status_step=200):
    assert M % batch_size == 0
    batch_Q = Q.copy()
    m = 1
    # SARSA main loop
    J = []
    delta_q = []
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
        batch_Q[s,a] = batch_Q[s,a] + alpha*(r + env.gamma*np.max(batch_Q[s_prime, :]) - batch_Q[s,a])
        if m % batch_size == 0:
            Q[s,a] = Q[s,a] + alpha*(r + env.gamma*np.max(batch_Q[s_prime, :]) - Q[s,a])
        if(m % status_step == 0):
            J.append(get_expected_avg_reward(env.P_mat, get_policy(Q), env.reward, env.gamma, env.mu))
            delta_q.append(np.linalg.norm(Q - Q_star, np.inf))
        # Setting next iteration
        m = m+1
        s = s_prime
        a = a_prime
    return Q, J, delta_q

"""
    Compare two different policies in terms of a distance measure
        @measure: the measure to be used for the comparison
        @pi: a policy row vector
        @pi_prime: a second policy row vector
        return the difference, according to the given measure, of the two policies
"""
def compare_policies(measure:DistanceMeasure, pi, pi_prime):
    return measure.compute_distance(pi, pi_prime)

