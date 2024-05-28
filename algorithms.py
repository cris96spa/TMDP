import numpy as np
from gymnasium import Env
from DiscreteEnv import DiscreteEnv
from TMDP import TMDP
from model_functions import *
from gymnasium.utils import seeding
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from ReplayBuffer import ReplayBuffer



"""
    Tabular softmax function. To reduce redundancy, the policy is parameterized using nA-1 parameters for each state, whereas the last one
    is computed considering that the softmax sums up to 1, allowing to implicitly determine last parameter.
    Args:
        - x (ndarray): parameter vector
        - temperature (float): temperature value
    return (ndarray): action probabilities associated to the given parameter vector
"""

"""def softmax_policy(x, t=1.0):
    e_x = np.exp((x-np.max(x))/t)
    return e_x / np.sum(e_x)"""

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
    done = False
    while not done:
        Q_old = Q.copy()
        for s in range(nS):
            for a in range(nA):
                Q[s,a] = r_s_a[s,a] + gamma * np.dot(P_mat[s,a,:], np.max(Q, axis=1))
        iterations += 1 
        epsilon = np.linalg.norm(Q - Q_old, np.inf)
        if epsilon <= threshold:
            done = True
    return {"Q": Q, iterations: iterations}

def bellman_optimal_q_tau(P_mat, xi, reward, gamma, tau, threshold=1e-6):
    nS, nA, _ = P_mat.shape
    r_s_a = compute_r_s_a(P_mat, reward)
    r_s_a_xi = compute_r_s_a(xi, reward)

    Q = np.zeros((nS, nA))
    Q_p = np.zeros((nS, nA))
    Q_xi = np.zeros((nS, nA))
    iterations = 0
    done = False
    while not done:
        Q_old = Q.copy()
        for s in range(nS):
            for a in range(nA):
                Q_p[s,a] = r_s_a[s,a] + gamma * np.dot(P_mat[s,a,:], np.max(Q, axis=1))
                Q_xi[s,a] = r_s_a_xi[s,a] + gamma * np.dot(xi, np.max(Q, axis=1))
                Q[s,a] = (1-tau)*Q_p[s,a] + tau*Q_xi[s,a]
        iterations += 1 
        epsilon = np.linalg.norm(Q - Q_old, np.inf)
        if epsilon <= threshold:
            done = True
    return {"Q": Q, "Q_p":Q_p, "Q_xi": Q_xi, "iterations": iterations}

def bellman_modified(P_mat, reward, gamma, tau, threshold=1e-6):
    nS, nA, _ = P_mat.shape
    r_s_a = compute_r_s_a(P_mat, reward)

    Q = np.zeros((nS, nA))
    iterations = 0
    done = False
    while not done:
        Q_old = Q.copy()
        for s in range(nS):
            for a in range(nA):
                Q[s,a] = tau * (r_s_a[s,a] + gamma*(2+tau)* np.dot(P_mat[s,a,:], np.max(Q, axis=1)))
        iterations += 1 
        epsilon = np.linalg.norm(Q - Q_old, np.inf)
        if epsilon <= threshold:
            done = True
    return {"Q": Q, "iterations": iterations}

def compute_gradient_q_tau(P_mat, xi, reward, mu, gamma, tau):
    nS, nA, _ = P_mat.shape

    Xi = np.tile(xi, (nA, nS)).T
    Xi = Xi.reshape((nS, nA, nS))
    P_mat_tau = (1-tau)*P_mat + tau*Xi

    res = bellman_optimal_q_tau(P_mat, xi, reward, gamma, tau)
    Q_p = res["Q_p"]
    Q_xi = res["Q_xi"]
    Q_tau = res["Q"]
    pi = get_policy(Q_tau)

    r_s_a = compute_r_s_a(P_mat, reward)
    r_s_a_xi = compute_r_s_a(xi, reward)
    
    d = compute_d_from_tau(mu, P_mat, xi, pi, gamma, tau)
    delta_r = r_s_a_xi - r_s_a

    grad_q = np.zeros_like(Q_p)
    sum_q = np.zeros(nS)
    for s in range(nS):
        for a in range(nA):
            sum_q[s] += pi[s,a]*(Q_xi[s,a] - Q_p[s,a])
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                grad_q[s,a] += (P_mat_tau[s,a,s_prime] + gamma/(1-gamma)*d[s_prime])*sum_q[s_prime]
            grad_q[s,a] = grad_q[s,a]*gamma + delta_r[s,a]
            

    return grad_q


"""
    Batch Q_learning algorithm implementation
    Args:
        - tmdp (DiscreteEnv): environment object
        - Q (ndarray): inizial state-action value function
        - episodes (int): number of episodes to be done
        - alpha (float): initial learning rate
        - batch_nS (int): nS of the batch
        - status_step (int): intermediate results flag. Used for the evaluation of the state action value function updates while learning
    return (dict): list of state action value functions updated at each status step, visit distributions, visit count, history of episodes
"""
def batch_q_learning(tmdp:TMDP, Q, episodes=5000, alpha=1., eps=0., status_step=5000, batch_nS=1):
    
    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []

    # Subset of the history, considering only the current batch
    batch = []
    trajectory_count = 0
    t = 0
    done = False

    nS, nA = Q.shape
    # Count the number of visits to each state
    visits = np.zeros(nS)
    visits_distr = np.zeros(nS)
    disc_visits = np.zeros(nS)
    disc_visits_distr = np.zeros(nS)
   
    if not eps:
        eps = min(1, alpha*2)
    
    # List of state action value functions updated at each status step
    Qs = []
    visits_distributions = []
    disc_visits_distributions = []
    
    dec_alpha = alpha
    dec_eps = eps
    

    for episode in range(episodes):
        # Q_learning main loop

        while True:
            s = tmdp.env.s
            # Pick an action according to the epsilon greedy policy
            a = eps_greedy(tmdp.env.s, Q, dec_eps, tmdp.env.allowed_actions[int(tmdp.env.s)])
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            his = {"state": s, "action": a, "reward": r, "next_state": s_prime, "flags": flags, "t": t}
            history.append(his)
            batch.append(his)
            
            # Setup next step
            tmdp.s = s_prime

            t += 1
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
                tmdp.reset()
                trajectory_count += 1

            if episode < episodes-1:    
                break
            else: # wait for the end of the trajectory to finish 
                if flags["done"]:
                    done = True
                    break

        if( (trajectory_count != 0 and trajectory_count % batch_nS == 0) or done):
            # Estimation of the number of batches per episode if all batches has the nS of the current one
            batch_per_episode = np.ceil(episode/len(batch))
            # Estimation of the total number of batches if all batches has the nS of the current one
            total_number_of_batches = np.ceil(episodes/len(batch))

            dec_alpha= max(0, alpha*(1-batch_per_episode/total_number_of_batches))
            dec_eps = max(0, eps*(1-batch_per_episode/total_number_of_batches)**2)

            ep_count = 0
            for ep in batch:
                
                s = ep["state"]
                a = ep["action"]
                r = ep["reward"]
                s_prime = ep["next_state"]
                flags = ep["flags"]
                t = ep["t"]

                # Policy improvement step
                a_prime = greedy(s_prime, Q, tmdp.env.allowed_actions[int(s_prime)])
                # Evaluation step
                Q[s,a] = Q[s,a] + dec_alpha*(r + tmdp.gamma*Q[s_prime, a_prime] - Q[s,a])

                visits[s] += 1
                disc_visits[s] += tmdp.gamma**ep_count

                visits_distr = visits/(np.sum(visits))
                disc_visits_distr = disc_visits/(np.sum(disc_visits))

                if flags["done"]:
                    ep_count = 0
                else:
                    ep_count += 1

            # Reset the batch
            batch = []
            trajectory_count = 0

        if episode % status_step == 0 or done:
            # Mid-result status update
            Qs.append(Q.copy())
            visits_distributions.append(visits_distr.copy())
            disc_visits_distributions.append(disc_visits_distr.copy())

    return {"Qs": Qs, "visits_distributions":visits_distributions, "visits": visits, "history": history, "disc_visits_distributions": disc_visits_distributions}

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
        - tmdp (DiscreteEnv): environment object
        - Qs (list): list of state action value functions
        - Q_star (ndarray): optimal state action value function
        - tau_prime (float): new teleportation probability, default 0 to evaluate the performance improvement in switching to the original problem
    return (dict): the computed metrics
"""
def compute_metrics(tmdp, Qs, Q_star,  visits_distributions, tau_prime=0., is_policy=False, temperature=1.0):
    Qs = Qs.copy()
    if is_policy:
        Qs.append(get_policy(Q_star))
    else:
        Qs.append(Q_star)
    d = compute_d(tmdp.env.mu, tmdp.P_mat_tau, get_policy(Q_star), tmdp.gamma)
    visits_distributions.append(d)
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
        if is_policy:
            pi = [softmax_policy(Q[s], temperature=temperature) for s in range(tmdp.env.nS)]
            pi = np.array(pi)
        else:
            pi = get_policy(Q)
        # Compute the expected discounted sum of rewards for the original problem and the simplified one with tau
        J.append(get_expected_avg_reward(tmdp.env.P_mat, pi, tmdp.env.reward, tmdp.gamma, tmdp.env.mu))
        J_tau.append(get_expected_avg_reward(tmdp.P_mat_tau, pi, tmdp.env.reward, tmdp.gamma, tmdp.env.mu))
        delta_J.append(J[-1] - J_tau[-1])

        # Compute the L_inf norm of the difference between the state action value function and the optimal one
        delta_Q.append(np.linalg.norm(Q - Q_star, np.inf))
        
        # Compute some model based metrics for the evaluation of the performance improvement
        d = compute_d(tmdp.env.mu, tmdp.P_mat_tau, pi, tmdp.gamma)
        delta = compute_delta(d, pi)
        U = get_state_action_nextstate_value_function(tmdp.P_mat_tau, tmdp.env.reward, tmdp.gamma, Q)
        rel_model_adv_hat = compute_relative_model_advantage_function_hat(tmdp.env.P_mat, tmdp.xi, U)
        dis_rel_model_adv = compute_discounted_distribution_relative_model_advantage_function_from_delta_tau(rel_model_adv_hat, delta, tmdp.tau, tau_prime)
        adv = dis_rel_model_adv/(1-tmdp.gamma)
        #Advantage term for the performance improvement lower bound
        adv_terms.append(adv)

        # Compute the dissimilarity penalization term for the performance improvement lower bound
        dq = get_sup_difference_Q(Q)
        de = get_expected_difference_transition_models(tmdp.env.P_mat, tmdp.xi, delta)
        dinf = get_sup_difference_transition_models(tmdp.env.P_mat, tmdp.xi)
        diss_pen = tmdp.gamma**2*(tmdp.tau - tau_prime)**2*dq*de*dinf/(2*(1-tmdp.gamma)**2)
        diss_terms.append(diss_pen)

        # Compute the lower bound for the performance improvement
        l_bounds.append(adv-diss_pen)

        # Compute the gradient of the expected discounted sum of rewards
        r_s_a_p = compute_r_s_a(tmdp.env.P_mat, tmdp.env.reward)
        r_s_a_xi = compute_r_s_a(tmdp.xi, tmdp.env.reward)
        
        Q_p = get_q_hat(tmdp.env.P_mat, r_s_a_p, tmdp.gamma, Q)
        Q_xi = get_q_hat(tmdp.xi, r_s_a_xi, tmdp.gamma, Q)
        grad = compute_grad_j(pi, Q_p, Q_xi, visits_distributions[i], tmdp.gamma)
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
