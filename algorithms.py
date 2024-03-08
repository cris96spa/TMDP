import numpy as np
from gym import Env
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
def Q_learning(env:DiscreteEnv, Q, episodes=5000, alpha=1., eps=0., status_step=5000, state_distribution=[]):

    nS, nA = Q.shape

    # Count the number of visits to each state
    visits = np.ones(nS)
    visit_distr = np.zeros(nS)
    
    # Probability transition matrix estimation
    counts = np.zeros((nS, nA, nS))
    P = np.zeros((nS, nA, nS))
    
    # Parametric Policy estimation ???
    param_policy = np.zeros((nS, nA))
   
    if not eps:
        eps = max(1, alpha*4)
    
    # List of state action value functions updated at each status step
    Qs = []
    visit_distributions = []
    
    dec_alpha = alpha
    dec_eps = eps

    for episode in range(episodes):
        # Q_learning main loop

        while True:
            s = env.s
            visits[s] += 1

            dec_eps = max(0, eps*(1-episode/episodes)**2)
            dec_alpha= max(0, alpha*(1-episode/episodes))

            if not len(state_distribution):
                visit_distr = visits/(np.sum(visits))

            # Pick an action according to the epsilon greedy policy
            a = eps_greedy(env.s, Q, dec_eps, env.allowed_actions[env.s.item()])
          
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  env.step(a)

            # Policy improvement step
            a_prime = greedy(s_prime, Q, env.allowed_actions[s_prime.item()])

            #print("Episode:", episode, " state:", s, " action:", a, " next state:",s_prime, " reward:",r, " next action:", a_prime, "epsilon:", eps, "alpha:", dec_alpha)
            # Evaluation step
            Q[s,a] = Q[s,a] + dec_alpha*(r + env.gamma*Q[s_prime, a_prime] - Q[s,a])

            # Update probability transition matrix estimation
            if not flags["teleport"]:
                
                counts[s, a, s_prime] += 1
                """print(counts)
                print(np.sum(counts[s_prime, :]))"""
                P[s, a, :] = counts[s, a, :]/np.sum(counts[s, a, :])
            
            # Setup next step
            env.s = s_prime
            
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
                env.reset()
            break

        if(episode % status_step == 0):
            Qs.append(Q.copy())
            if not len(state_distribution):
                visit_distributions.append(visit_distr)
            else:  
                visit_distributions.append(state_distribution)
                
    if(episode % status_step != 0):
        Qs.append(Q.copy())
        if not len(state_distribution):
            visit_distributions.append(visit_distr)
        else:  
            visit_distributions.append(state_distribution)

    return {"Qs": Qs, "visit_distributions":visit_distributions, "P":P, "counts":counts, "visits": visits}


"""
    Compute some metrics for the evaluation of the performance improvement
    The metrics computed are:
        - J: list of expected discounted sum of rewards for the original problem over the state action value functions
        - J_tau: list of expected discounted sum of rewards for the simplified problem with tau over the state action value functions
        - delta_J: list of differences between J and J_tau over the state action value functions
        - delta_Q: list of L_inf norm of the differences between the state action value function and the optimal one 
        - adv_terms: list of advantage terms for the performance improvement lower bound over the state action value functions
        - diss_terms: list of dissimilarity penalization terms for the performance improvement lower bound over the state action value functions
        - l_bounds: list of lower bounds for the performance improvement over the state action value functions
    Args:
        - env (DiscreteEnv): environment object
        - Qs (list): list of state action value functions
        - Q_star (ndarray): optimal state action value function
        - tau_prime (float): new teleportation probability, default 0 to evaluate the performance improvement in switching to the original problem
    return (dict): the computed metrics
"""
def compute_metrics(env, Qs, Q_star,  visit_distributions, tau_prime=0.):
    Qs = Qs.copy()
    Qs.append(Q_star)
    d = compute_d(env.mu, env.P_mat_tau, get_policy(Q_star), env.gamma)
    visit_distributions.append(d)
    metrics = {}
    J = []
    J_tau = []
    delta_J = []
    delta_Q = []
    grad_J = []
    adv_terms = []
    diss_terms = []
    l_bounds = []
    for i, Q in enumerate(Qs):
        pi = get_policy(Q)
        # Compute the expected discounted sum of rewards for the original problem and the simplified one with tau
        J.append(get_expected_avg_reward(env.P_mat, pi, env.reward, env.gamma, env.mu))
        J_tau.append(get_expected_avg_reward(env.P_mat_tau, pi, env.reward, env.gamma, env.mu))
        delta_J.append(J[-1] - J_tau[-1])

        # Compute the L_inf norm of the difference between the state action value function and the optimal one
        delta_Q.append(np.linalg.norm(Q - Q_star, np.inf))
        
        # Compute some model based metrics for the evaluation of the performance improvement
        d = compute_d(env.mu, env.P_mat_tau, pi, env.gamma)
        delta = compute_delta(d, pi)
        U = get_state_action_nextstate_value_function(env.P_mat_tau, env.reward, env.gamma, Q)
        rel_model_adv_hat = compute_relative_model_advantage_function_hat(env.P_mat, env.xi, U)
        dis_rel_model_adv = compute_discounted_distribution_relative_model_advantage_function_from_delta_tau(rel_model_adv_hat, delta, env.tau, tau_prime)
        adv = dis_rel_model_adv/(1-env.gamma)
        #Advantage term for the performance improvement lower bound
        adv_terms.append(adv)

        # Compute the dissimilarity penalization term for the performance improvement lower bound
        dq = get_sup_difference_q(Q)
        de = get_expected_difference_transition_models(env.P_mat, env.xi, delta)
        dinf = get_sup_difference_transition_models(env.P_mat, env.xi)
        diss_pen = env.gamma**2*(env.tau - tau_prime)**2*dq*de*dinf/(2*(1-env.gamma)**2)
        diss_terms.append(diss_pen)

        # Compute the lower bound for the performance improvement
        l_bounds.append(adv-diss_pen)

        # Compute the gradient of the expected discounted sum of rewards
        r_s_a_p = compute_r_s_a(env.P_mat, env.reward)
        r_s_a_xi = compute_r_s_a(env.xi, env.reward)
        Q_p = get_q_hat(env.P_mat, r_s_a_p, env.gamma, Q)
        Q_xi = get_q_hat(env.xi, r_s_a_xi, env.gamma, Q)
        grad = compute_grad_j(pi, Q_p, Q_xi, visit_distributions[i], env.gamma)
        grad_J.append(grad)

    metrics["J"] = J
    metrics["J_tau"] = J_tau
    metrics["grad_J"] = grad_J
    metrics["delta_J"] = delta_J
    metrics["delta_Q"] = delta_Q

    metrics["adv_terms"] = adv_terms
    metrics["diss_terms"] = diss_terms
    metrics["l_bounds"] = l_bounds

    return metrics
