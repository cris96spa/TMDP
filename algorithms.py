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


seed = None
#seed = 73111819126096741712253486776689977811

np_random, seed = seeding.np_random(seed)
def get_current_seed():
    return seed

print("Current seed for result reproducibility: {}".format(seed))

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
    if np_random.random() <= eps:
        actions = np.where(allowed_actions)
        # Extract indices of allowed actions
        actions = actions[0]
        # pick a uniformly random action
        a = np_random.choice(actions, p=(np.ones(len(actions))/len(actions)))
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
    Softmax action selection
    Args:
        - s (int): current state
        - theta (ndarray): parameter vector
        - allowed_actions (ndarray): array of allowed actions
        - t (float): temperature value
    return (int): the action to be taken
"""

"""
    Get the softmax probability associated to the parameter vector of a single state
    Args:
        - x (ndarray): parameter vector of shape [nA-1]
        - temperature (float): temperature value
    return (ndarray): the softmax policy probabilities associated to a single state
"""
def softmax_policy(x, temperature=1.0, redundant=True):
    # Apply the temperature scale and consider an implicit parameter for last action of 1
    param = x/temperature
    if not redundant:
        param = np.append(param, 1)
    exp = np.exp(param - np.max(param))
    return exp / np.sum(exp)

"""
    Get the overall softmax policy from parameter matrix
    Args:
        - x (ndarray): parameter matrix of shape [nS, nA-1]
        - temperature (float): temperature value
    return (ndarray): the overall softmax policy
"""
def get_softmax_policy(x, temperature=1.0, redundant=True):
    # Apply the temperature scale and consider an implicit parameter for last action of 1
    nS, nA = x.shape
    if not redundant:
        nA += 1
    exp = np.array([])
    for s in range(nS):
        exp = np.append(exp, softmax_policy(x[s], temperature=temperature))
    return exp.reshape((nS, nA))


def select_action(prob):
    a = np_random.choice(len(prob), p=prob)
    return int(a)

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


def curriculum_AC(tmdp:TMDP, Q, episodes=5000, alpha=.25, alpha_pol=.1, status_step=50000, 
                  batch_nS=1, temperature=1.0, lam=0., biased=True, epochs=1, use_delta_Q=False,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    nS, nA = Q.shape

    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []
    # Subset of the history, considering only the current batch
    batch = []
    # Subset of the batch, considering only the current trajectory
    traj = []

    rewards = []
    reward_records = []

    t = 0 # episode in batch counter
    k = 0 # episode in trajectory counter
    ep = 0 # overall episode counter

    teleport_count = 0

    done = False 
    terminated = False

    #Policy parameter vector, considering nA parameters for each state
    # Random initialization
    theta = np.zeros((nS, nA))
    #theta = np.zeros((nS, nA))
    theta_ref = np.zeros((nS, nA))

    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    convergence_t = 0
    
    # Curriculum parameters
    alpha_star = 1
    tau_star = 1
  
    # Mid-term results
    Qs = []
    thetas = []

    e = np.zeros_like(Q)
    decay = 1
    temp_decay = 1
    grad_log_policy = np.zeros(nA)

    # Tensor conversion
    tensor_mu = torch.tensor(tmdp.env.mu, dtype=torch.float32).to(device)
    tensor_P_mat = torch.tensor(tmdp.env.P_mat, dtype=torch.float32).to(device)
    tensor_xi = torch.tensor(tmdp.xi, dtype=torch.float32).to(device)

    d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
    for episode in range(episodes): # Each episode is a single time step

        pi = get_softmax_policy(theta, temperature=temperature*temp_decay)
        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = select_action(pi[s])
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            a_prime = select_action(pi[s_prime])
            flags["terminated"] = terminated
            

            if not flags["teleport"]:
                sample = (s, a, r, s_prime, a_prime, flags, t, k)
                traj.append(sample)
                rewards.append(r)
                k += 1
                t += 1
                if flags["done"]:
                    tmdp.reset()
                    # add the current trajectory
                    batch.append(traj)
                    history.append(traj)
                    # Reset the trajectory
                    traj = []
                    k = 0
            else:
                teleport_count += 1
                if len(traj) > 0:
                    batch.append(traj)
                    history.append(traj)
                    traj = []
                    k = 0

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                flags["terminated"] = terminated
                batch.append(traj)
                history.append(traj)
                traj = []
                k = 0
                break # temporary ending condition To be Removed
                if flags["done"]:
                    done = True
                    break

        # Processing the batch
        if( (len(batch) != 0 and len(batch) % batch_nS == 0) or done or terminated):
            # Extract previous policy for future comparison
            
            # Iterate over trajectories in the batch
            for _ in range(epochs):
                for trajectory in batch:
                    # Iterate over samples in the trajectory
                    for j, sample in enumerate(trajectory):
                        s, a, r, s_prime, a_prime, flags, t, k  = sample
                        
                        #a_prime = greedy(s_prime, Q, tmdp.env.allowed_actions[int(s_prime)]) # Q must be done on policy
                        td_error = alpha*decay*(r + tmdp.gamma*Q[s_prime, a_prime] - Q[s,a])

                        # Eligibility traces
                        e[s,a] = 1 # Freequency heuristic with saturation
                        if lam == 0:
                            Q[s,a] += e[s,a]*td_error
                        else:
                            for s_1 in range(nS):
                                for a_1 in range(nA):
                                    Q[s_1,a_1] +=  e[s_1,a_1]*td_error
                        e = tmdp.gamma*lam*e # Recency heuristic decay
                        
                        # Get current policy
                        pi_ref = get_softmax_policy(theta_ref, temperature=temperature*temp_decay)
                        V = compute_V_from_Q(Q, pi)
                        U[s,a,s_prime] += alpha*decay*(r + tmdp.gamma*V[s_prime] - U[s,a,s_prime])

                        # Policy Gradient
                        grad_log_policy = -pi_ref[s]
                        grad_log_policy[a] += 1 # Sum 1 for the taken action
                        grad_log_policy = grad_log_policy/(temperature*temp_decay)
                        theta_ref[s] += alpha_pol*decay*grad_log_policy*(Q[s,a] - V[s]) # Policy parameters update
                        ep += 1 # Increase the episode counter
            
            r_sum = sum(rewards)

            ################################ Bound evaluation ################################
            s_time = time.time()
            # Get policies
            pi_ref = get_softmax_policy(theta_ref, temperature=temperature*temp_decay)
            pi = get_softmax_policy(theta, temperature=temperature*temp_decay)
            
            # Tensor conversion
            tensor_pi_ref = torch.tensor(pi_ref, dtype=torch.float32).to(device)
            tensor_pi = torch.tensor(pi, dtype=torch.float32).to(device)
            tensor_Q = torch.tensor(Q, dtype=torch.float32).to(device)
            tensor_U = torch.tensor(U, dtype=torch.float32).to(device)
            
            # Compute advantages
            rel_pol_adv = compute_relative_policy_advantage_function(tensor_pi_ref, tensor_pi, tensor_Q)
            d = compute_d_from_tau(tensor_mu, tensor_P_mat, tensor_xi, tensor_pi, tmdp.gamma, tmdp.tau)
            pol_adv = compute_expected_policy_advantage(rel_pol_adv, d) 
            
            if use_delta_Q:
                delta_U = get_sup_difference(tensor_Q)
            else:
                delta_U = get_sup_difference(tensor_U) 
            if delta_U == 0:
                delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                
            # Compute distance metrics
            d_inf_pol = get_d_inf_policy(tensor_pi, tensor_pi_ref)
            d_exp_pol = get_d_exp_policy(tensor_pi, tensor_pi_ref, d)
            
            if( tmdp.tau > 0): # Not yet converged to the original problem
                delta = compute_delta(d, tensor_pi)
                rel_model_adv = compute_relative_model_advantage_function(tensor_P_mat, tensor_xi, tensor_U)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)
                
                d_exp_model = get_d_exp_model(tensor_P_mat, tensor_xi, delta)
            
            else: # Converged to the original problem
                model_adv = 0
                d_exp_model = 0

            
            # Compute optimal values
            optimal_pairs = get_teleport_bound_optimal_values(pol_adv, model_adv, delta_U,
                                                            d_inf_pol, d_exp_pol, d_inf_model,
                                                            d_exp_model, tmdp.tau, tmdp.gamma, biased=biased)
            teleport_bounds = []
            for alpha_prime, tau_prime in optimal_pairs:
                bound = compute_teleport_bound(alpha_prime, tmdp.tau, tau_prime, pol_adv, model_adv,
                                                tmdp.gamma, d_inf_pol, d_inf_model,
                                                d_exp_pol, d_exp_model, delta_U, biased=biased)
                teleport_bounds.append(bound)
            
            # Get the optimal values
            alpha_star, tau_star = get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds)

            print(optimal_pairs)
            if alpha_star != 0 or tau_star != 0:

                print("Alpha*: {} tau*: {} Episode: {} length: {} #teleports:{}".format(alpha_star, tau_star, episode, len(rewards),teleport_count))
            else:
                print("No updates performed, episode: {} length: {} #teleports:{}".format(episode, len(rewards),teleport_count))
            if r_sum > 0:
                print("Got not null reward {}!".format(r_sum))

            if tau_star == 0 and tmdp.tau != 0:
                print("Converged to the original problem, episode {}".format(episode))
                convergence_t = episode
                tmdp.update_tau(tau_star)
            elif tmdp.tau > 0:
                tmdp.update_tau(tau_star)

            if alpha_star != 0:
                theta = alpha_star*theta_ref + (1-alpha_star)*theta

            decay = max(1e-8, 1-(ep)/(episodes))
            temp_decay = temperature + (1e-3 - temperature)*(ep/episodes)

            # Reset the batch
            batch = []
            reward_records.append(r_sum)
            rewards = []
            t = 0
            teleport_count = 0
            e_time = time.time()
            print("Time for bound evaluation: ", e_time - s_time)
        if episode % status_step == 0 or done or terminated:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(np.copy(Q))
            thetas.append(np.copy(theta))

    return {"Qs": Qs, "history": history, "thetas": thetas, "reward_records": reward_records}

def curriculum_PPO_test(tmdp:TMDP, Q, episodes=5000, alpha=.25, alpha_pol=.1, status_step=50000, 
                  batch_nS=1, temperature=1.0, lam=0., biased=True, epochs=1):
    nS, nA = Q.shape

    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []
    # Subset of the history, considering only the current batch
    batch = []
    # Subset of the batch, considering only the current trajectory
    traj = []

    rewards = []
    reward_records = []

    t = 0 # episode in batch counter
    k = 0 # episode in trajectory counter
    ep = 0 # overall episode counter

    teleport_count = 0

    done = False 
    terminated = False

    #Policy parameter vector, considering nA parameters for each state
    # Random initialization
    theta = tmdp.env.np_random.random((nS, nA))
    #theta = np.zeros((nS, nA))
    theta_ref = tmdp.env.np_random.random((nS, nA))

    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    convergence_t = 0
    
    # Curriculum parameters
    alpha_star = 1
    tau_star = 1
  
    # Mid-term results
    Qs = []
    thetas = []

    e = np.zeros_like(Q)
    decay = 1
    temp_decay = 1
    grad_log_policy = np.zeros(nA)
    for episode in range(episodes): # Each episode is a single time step
        
        pi = get_softmax_policy(theta, temperature=temperature*temp_decay)
        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = select_action(pi[s])
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)

            flags["terminated"] = terminated
            if not flags["teleport"]:
                sample = (s, a, r, s_prime, flags, t, k)
                traj.append(sample)
                rewards.append(r)
                k += 1
                t += 1
                if flags["done"]:
                    tmdp.reset()
                    # add the current trajectory
                    batch.append(traj)
                    history.append(traj)
                    # Reset the trajectory
                    traj = []
                    k = 0
            else:
                teleport_count += 1
            # Reset the environment if a terminal state is reached
            if flags["done"]:
                tmdp.reset()
                # add the current trajectory
                batch.append(traj)
                history.append(traj)
                # Reset the trajectory
                traj = []
                k = 0

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                sample["flags"]["terminated"] = terminated
                batch.append(traj)
                history.append(traj)
                traj = []
                k = 0
                break # temporary ending condition To be Removed
                if flags["done"]:
                    done = True
                    break

        # Processing the batch
        if( (len(batch) != 0 and len(batch) % batch_nS == 0) or done or terminated):
            # Extract previous policy for future comparison
            
            # Iterate over trajectories in the batch
            for _ in range(epochs):
                for trajectory in batch:
                    # Iterate over samples in the trajectory
                    for j, sample in enumerate(trajectory):
                        # Recover sample data
                        s, a, r, s_prime, flags, t, k  = sample

                        if not flags["teleport"]:
                            a_prime = greedy(s_prime, Q, tmdp.env.allowed_actions[int(s_prime)])
                            td_error = alpha*decay*(r + tmdp.gamma*Q[s_prime, a_prime] - Q[s,a])

                            # Eligibility traces
                            e[s,a] = 1 # Freequency heuristic with saturation
                            if lam == 0:
                                Q[s,a] += e[s,a]*td_error
                            else:
                                for s_1 in range(nS):
                                    for a_1 in range(nA):
                                        Q[s_1,a_1] +=  e[s_1,a_1]*td_error
                            e = tmdp.gamma*lam*e # Recency heuristic decay
                            
                            # Get current policy
                            pi_ref = get_softmax_policy(theta_ref, temperature=temperature*decay)
                            V = compute_V_from_Q(Q, pi_ref)
                            U[s,a,s_prime] =  r + tmdp.gamma*V[s_prime]

                            # Policy Gradient
                            grad_log_policy = -pi_ref[s]
                            grad_log_policy[a] += 1 # Sum 1 for the taken action
                            grad_log_policy = grad_log_policy/(temperature*decay)
                            theta_ref[s] += alpha_pol*decay*grad_log_policy*(Q[s,a] - V[s]) # Policy parameters update
                        else:
                            teleport_count += 1
                        ep += 1 # Increase the episode counter
            
            r_sum = sum(rewards)

            # Bound evaluation
            if( tmdp.tau > 0):
                # Get policies
                pi_ref = get_softmax_policy(theta_ref, temperature=temperature*decay)
                pi = get_softmax_policy(theta, temperature=temperature*decay)

                # Compute advantages
                rel_pol_adv = compute_relative_policy_advantage_function(pi_ref, pi, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)

                delta_U = get_sup_difference_U(U)
                if delta_U == 0:
                    delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                
                # Compute distance metrics
                d_inf_pol = get_d_inf_policy(pi, pi_ref)
                d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
                d_exp_pol = get_d_exp_policy(pi, pi_ref, d)
                d_exp_model = get_d_exp_model(tmdp.env.P_mat, tmdp.xi, delta)

                # Compute optimal values
                optimal_pairs = get_teleport_bound_optimal_values(pol_adv, model_adv, delta_U,
                                                                d_inf_pol, d_exp_pol, d_inf_model,
                                                                d_exp_model, tmdp.tau, tmdp.gamma, biased=biased)
                teleport_bounds = []
                for alpha_prime, tau_prime in optimal_pairs:
                    bound = compute_teleport_bound(alpha_prime, tmdp.tau, tau_prime, pol_adv, model_adv,
                                                    tmdp.gamma, d_inf_pol, d_inf_model,
                                                    d_exp_pol, d_exp_model, delta_U, biased=biased)
                    teleport_bounds.append(bound)
                
                # Get the optimal values
                alpha_star, tau_star = get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds)

                print(optimal_pairs)
                print("Alpha*: {} tau*: {} Episode: {} length: {} #teleports:{}".format(alpha_star, tau_star, episode, len(rewards),teleport_count))
                if r_sum > 0:
                    print("Got not null reward {}!".format(r_sum))
                tmdp.update_tau(tau_star)
                theta = alpha_star*theta_ref + (1-alpha_star)*theta
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    # Move to the learned policy for fine tuning
                    theta = theta_ref
                    print(Q)
                    print(theta)
            else:
                theta = theta_ref
                print("Episode: {} length: {}".format(episode, len(rewards)))
                if r_sum > 0:
                    print("Got not null reward {}!".format(r_sum))
                
            decay = max(1e-6, 1-(ep)/(episodes))
            temp_decay = temperature + (1 - temperature)*(ep/episodes)

            # Reset the batch
            batch = []
            reward_records.append(r_sum)
            rewards = []
            t = 0
            teleport_count = 0

        if episode % status_step == 0 or done or terminated:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(np.copy(Q))
            thetas.append(np.copy(theta))

    return {"Qs": Qs, "history": history, "thetas": thetas, "reward_records": reward_records}

