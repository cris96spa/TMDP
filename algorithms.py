import numpy as np
from gym import Env
from distanceMeasure import *
from DiscreteEnv import DiscreteEnv
from scipy.special import softmax
from model_functions import *


"""
    Compute the optimal state action value function using the Bellman optimality operator as Q*(s,a) = R(s,a) + gamma *sum_{s' in S}P(s'|s,a) * max_{a' in A}Q*(s',a')
    Args:    
        - P_mat (ndarray): transition probability matrix [nS, nA, nS]
        - reward (ndarray): reward function [nS, nA]
        - gamma (float): discount factor
        - threshold (float): convergence threshold
    return (dict): the optimal state action value function and the number of iterations needed to converge
"""
def bellman_optimal_q(P_mat, reward, gamma, threshold=1e-6):
    nS, nA, _ = P_mat.shape
    r_s_a = compute_r_s_a(P_mat, reward)
    Q = np.zeros((nS, nA))
    iterations = 0
    loop = True
    while loop:
        Q_old = Q.copy()
        for s in range(nS):
            for a in range(nA):
                Q[s,a] = r_s_a[s,a] + gamma * np.dot(P_mat[s,a,:], np.max(Q, axis=1))
        iterations += 1 
        epsilon = np.linalg.norm(Q - Q_old, np.inf)
        if epsilon <= threshold:
            loop = False
    return {"Q": Q, iterations: iterations}

"""
    Epsilon greedy action selection
    Args:
        - s (int): current state
        - Q (ndarray): state action value function
        - eps (float): exploration rate
        - allowed_actions (ndarray): array of allowed actions
    return (int): the action to be taken
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
    Greedy action selection
    Args:
        - s (int): current state
        - Q (ndarray): state action value function
        - allowed_actions (ndarray): array of allowed actions
    return (int): the action to be taken
"""
def greedy(s, Q, allowed_actions):
    # Extract the Q function for the given state
    Q_s = Q[s, :].copy()
    # Set to -inf the state action value function of not allowed actions
    Q_s[allowed_actions == 0] = -np.inf
    a = np.argmax(Q_s)
    return a

"""
    SARSA algorithm implementation.
    Args:
        - env (DiscreteEnv): environment object
        - s (int): current state
        - a (int): action to be taken
        - Q (ndarray): current state-action value function
        - M (int): number of iterations to be done
    return (ndarray): the state action value function under the pseudo-optimal policy found
"""
def SARSA(env:DiscreteEnv, s, a, Q, M=5000):
    episods = 1
    # Learning rate initialization
    alpha = (1- episods/M)
    # epsilon update
    eps = (1 - episods/M)**2

    # SARSA main loop
    while episods < M:
        # Perform a step in the environment, picking action a
        s_prime, r, flags, p = env.step(a)

        # Policy improvement step
        # N.B. allowed action is not present in the Env object, must be managed
        a_prime = eps_greedy(s_prime, Q, eps, env.allowed_actions[s_prime.item()])
        # Evaluation step
        Q[s,a] = Q[s,a] + alpha*(r + env.gamma*Q[s_prime, a_prime] - Q[s,a])

        # Setting next iteration
        env.s = s_prime
        a = a_prime
        if flags["done"]: #or flags["teleport"]:
            env.reset()
            a = eps_greedy(env.s, Q, eps, env.allowed_actions[env.s.item()])
            episods += 1
            eps = max(0,(1 - episods/M)**2)
            alpha= max(0, (1 - episods/M))
    return Q

"""
    Q_learning algorithm implementation
        - env (DiscreteEnv): environment object
        - Q (ndarray): inizial state-action value function
        - episodes (int): number of episodes to be done
        - alpha (float): initial learning rate
        - status_step (int): intermediate results flag. Used for the evaluation of the state action value function updates while learning
"""
def Q_learning_2(env:DiscreteEnv, Q, episodes=5000, alpha=0.5, status_step=5000):

    # Initialize the step counter
    nS, nA = Q.shape
    # Count the number of visits to each state
    visits = np.zeros(nS)

    # Learning rate decay parameters
    min_alpha = 0.01

    # Exploration rate
    eps = 1
    # Epsilon decay parameters
    min_epsilon = 0.01
    
    # List of state action value functions updated at each status step
    Qs = []
    # Pick the first action to be taken
    
    for episode in range(episodes):
        # Q_learning main loop

        while True:
            s = env.s
            visits[s] += 1
            a = eps_greedy(env.s, Q, eps, env.allowed_actions[env.s.item()])
            
            # Current state visit counter
            visits[env.s]+= 1
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  env.step(a)

            # Policy improvement step
            # N.B. allowed action is not present in the Env object, must be managed
            a_prime = greedy(s_prime, Q, env.allowed_actions[s_prime.item()])

            #print("Episode:", episode, " state:", s, " action:", a, " next state:",s_prime, " reward:",r, " next action:", a_prime, "epsilon:", eps, "alpha:", dec_alpha)
            # Evaluation step
            Q[s,a] = Q[s,a] + alpha*(r + env.gamma*Q[s_prime, a_prime] - Q[s,a])
            
            """current_q_value = Q[env.s, a]
            best_next_q_value = np.max(Q[s_prime, :])
            new_q_value = (1 - alpha) * current_q_value + alpha * (r + env.gamma * best_next_q_value)
            Q[env.s, a] = new_q_value"""
            
            # Setup next step
            env.s = s_prime
            
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"] or flags["teleport"]:
                env.reset()
                a = eps_greedy(env.s, Q, eps, env.allowed_actions[env.s.item()])
                eps = max(min_epsilon, (1-episode/episodes)**2)
                alpha= max(min_alpha, (1-episode/episodes))
                break

        if(episode % status_step == 0):
                Qs.append(Q.copy())
    if(episode % status_step != 0):
                Qs.append(Q.copy())

    return {"Qs": Qs, "visits": visits}


def compute_metrics(env, Qs, Q_star):
    metrics = {}
    J = []
    J_tau = []
    delta_Q = []
    delta_J = []
    l_bounds = []
    for Q in Qs:
        J.append(get_expected_avg_reward(env.P_mat, get_policy(Q), env.reward, env.gamma, env.mu))
        J_tau.append(get_expected_avg_reward(env.P_mat_tau, get_policy(Q), env.reward, env.gamma, env.mu))
        delta_Q.append(np.linalg.norm(Q - Q_star, np.inf))
        delta_J.append(J[-1] - J_tau[-1])
        d = compute_d(env.mu, env.P_mat_tau, get_policy(Q), env.gamma)
        delta = compute_delta(d, get_policy(Q))
        U = get_state_action_nextstate_value_function(env.P_mat_tau, env.reward, env.gamma, Q)
        rel_model_adv_hat = compute_relative_model_advantage_function_hat(env.P_mat, env.xi, U)
        dis_rel_model_adv = compute_discounted_distribution_relative_model_advantage_function_from_delta_tau(rel_model_adv_hat, delta, env.tau, 0.)
        p
        #model_adv = get_model_advantage_function(env.P_mat_tau, env.reward, env.tmdp.gamma, Q)

    metrics["J"] = J
    metrics["J_tau"] = J_tau
    metrics["delta_Q"] = delta_Q
    metrics["delta_J"] = delta_J
    metrics["l_bounds"] = l_bounds
    return metrics

def Q_learning(env:Env, s, a, Q, Q_star, M=5000, alpha=0., status_step=200, debug=False, main_p=True):
    m = 0
    J_main_p = []
    J_curr_p = []
    delta_q = []
    delta_J = []
    l_ = []
    nS, nA = Q.shape
    visits = np.zeros(nS)
    dec_alpha = np.ones(nS)*alpha
    # Q_learning main looph
    while m < M:
        # Learning rate initialization
        visits[s] = visits[s]+1
        # epsilon update
        eps = (1 - m/M)**2
        # Perform a step in the environment, picking action a
        s_prime, r, d, p =  env.step(a, debug=debug)

        # Policy improvement step
        # N.B. allowed action is not present in the Env object, must be managed
        a_prime = eps_greedy(s_prime, Q, eps, env.allowed_actions[s_prime.item()])

        #print("Step:", m, " state:", s, " action:", a, " next state:",s_prime, " reward:",r, " next action:", a_prime)
        # Evaluation step
        Q[s,a] = Q[s,a] + dec_alpha[s]*(r + env.gamma*np.max(Q[s_prime, :]) - Q[s,a])

        #Compute performance metrics for evaluation purposes
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
            r_s_a = compute_r_s_a(env.P_mat_tau, env.reward)
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
        - env: environment object
        - s: current state
        - a: first action to be taken
        - Q: current state-action value function
        - M: number of iterations to be considered
        - status: intermediate results flag
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
        - measure: the measure to be used for the comparison
        - pi: a policy row vector
        - pi_prime: a second policy row vector
        return the difference, according to the given measure, of the two policies
"""
def compare_policies(measure:DistanceMeasure, pi, pi_prime):
    return measure.compute_distance(pi, pi_prime)

