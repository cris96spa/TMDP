import numpy as np
from gymnasium import Env
from DiscreteEnv import DiscreteEnv
from TMDP import TMDP
from model_functions import *
from gymnasium.utils import seeding
import torch
import torch.nn as nn
from torch.nn import functional as F
from algorithms import *

def curriculum_MPI_2(tmdp:TMDP, Q, episodes=5000, alpha=.25, alpha_pol=.1, status_step=50000, epochs=10, batch_nS=1, temperature=1.0, lam=0., biased=True):
    nS, nA = Q.shape

    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []
    # Subset of the history, considering only the current batch
    batch = []
    # Subset of the batch, considering only the current trajectory
    traj = []

    t = 0 # batch counter
    k = 0 # trajectory counter
    done = False 
    terminated = False

    #Policy parameter vector, considering nA parameters for each state
    # Random initialization
    theta = tmdp.env.np_random.random((nS, nA))
    #theta = np.zeros((nS, nA))
    theta_ref = np.copy(theta)

    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    V = np.zeros(nS)
    # Hyperparameters decay
    dec_alpha = alpha
    dec_alpha_pol = alpha_pol
    final_temp = 1e-5 # Final temperature
    temp = temperature # Current temperature
    convergence_t = 0
    
    # Curriculum parameters
    alpha_star = dec_alpha
    tau_star = 1
  
    # Mid-term results
    Qs = []
    thetas = []

    e = np.zeros_like(Q)
    l = lam
    for episode in range(episodes): # Each episode is a single time step
        
        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = select_action(tmdp.env.s, theta, temperature=temp)
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            flags["terminated"] = terminated
            sample = {"state": s, "action": a, "reward": r, "next_state": s_prime, "flags": flags, "t": t, "k": k}
            traj.append(sample)
            # Setup next step
            tmdp.env.s = s_prime
            k += 1
            t += 1
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
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
            
            pi_old = get_softmax_policy(theta, temperature=temp)
            theta_ref = np.copy(theta)
            compute_advantages(batch, V, l, tmdp.gamma)# Compute advantages during trajectories
            # Iterate over trajectories in the batch
            for _ in range(epochs):
                for trajectory in batch:
                    # Iterate over samples in the trajectory
                    for j, sample in enumerate(trajectory):
                        s = sample["state"]
                        a = sample["action"]
                        r = sample["reward"]
                        s_prime = sample["next_state"]
                        flags = sample["flags"]
                        t = sample["t"]
                        adv = sample["adv"]
                        # learning parameters decay
                        if tmdp.tau != 0:
                            temp = temperature 
                            dec_alpha= alpha
                            dec_alpha_pol = alpha_pol
                        else:
                            temp = temperature + (final_temp - temperature)*((episode - convergence_t)/(episodes-convergence_t))
                            dec_alpha= max(1e-5, alpha*(1 - (episode-convergence_t)/(episodes-convergence_t)))
                            dec_alpha_pol = max(1e-5, alpha_pol*(1 - (episode-convergence_t)/(episodes-convergence_t)))
                    
                        pi = get_softmax_policy(theta_ref, temperature=temp)
                        # Computing the gradient of the log policy
                        grad_log_policy = -pi[s]

                        grad_log_policy[a] += 1 
                        """# For the taken action, a + 1 is considerd if it is not the explicit one
                        if a < nA-1:
                            grad_log_policy[a] += 1
                        else:
                            # gradient set to 0 for the explicit action
                            grad_log_policy[a] = 0"""

                        grad_log_policy = grad_log_policy/temperature
                        """if tmdp.tau == 0.:
                            print("Fine tuning the policy")
                            print("Gradient: ", grad_log_policy)
                            print("dec_alpha: ", dec_alpha)
                            print("Q: ", Q[s,a])
                            print("V: ", V[s])
                            print(dec_alpha*grad_log_policy*(Q[s,a] - V[s]))"""
                        theta_ref[s] += dec_alpha_pol*grad_log_policy*adv # Policy parameters update
                        
                        # Evaluation step
                        td_error = dec_alpha*(r + tmdp.gamma*V[s_prime] - V[s])
                        V[s] = V[s] + td_error
                        U[s,a,s_prime] = U[s,a,s_prime] + dec_alpha*(r + tmdp.gamma*V[s_prime] - U[s,a,s_prime])
                
            # Reset the batch
            batch = []
            t = 0

            # Bound evaluation
            if( tmdp.tau > 0):
                l = 0.
                rel_pol_adv = compute_relative_policy_advantage_function(pi, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)
                delta_Q = get_sup_difference_Q(Q)

                delta_U = get_sup_difference_U(U)
                if delta_U == 0:
                    delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                    
                d_inf_pol = get_d_inf_policy(pi, pi_old)
                d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
                d_exp_pol = get_d_exp_policy(pi, pi_old, d)
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
                
                alpha_star, tau_star = get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds)

                print(optimal_pairs)
                print(teleport_bounds)
                print("Updating the policy with alpha_star: ", alpha_star, "tau_star: ",tau_star)
                tmdp.update_tau(tau_star)
                theta = alpha_star*theta_ref + (1-alpha_star)*theta
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    # Set lambda parameter for eligibility traces for the fine tuning of the policy
                    l=lam
                    print(Q)
                    print(theta)
                    conv_Q = np.copy(Q)
                    conv_theta = np.copy(theta)
                    theta_ref = theta
            else:
                theta = theta_ref
                print("Working on the original problem, episode {}".format(episode))

        if episode % status_step == 0 or done:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(np.copy(Q))
            thetas.append(np.copy(theta))

    return {"Qs": Qs, "history": history, "thetas": thetas}#, "conv_Q": conv_Q, "conv_theta": conv_theta, "convergence_t": convergence_t}


def curriculum_PPO_2(tmdp:TMDP, Q, episodes=5000, epochs=10, alpha=.25, alpha_pol=.125, status_step=50000, batch_nS=1, temperature=1.0, lam=0., eps=0.2, biased=True):
    
    nS, nA = Q.shape
    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []
    # Subset of the history, considering only the current batch
    batch = []
    # Subset of the batch, considering only the current trajectory
    traj = []

    t = 0 # batch counter
    k = 0 # trajectory counter
    done = False 
    terminated = False

    #Policy parameter vector, considering nA parameters for each state
    # Random initialization
    theta = tmdp.env.np_random.random((nS, nA))
    #theta = np.zeros((nS, nA))
    theta_ref = np.copy(theta)

    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    V = np.zeros(nS)

    # Hyperparameters decay
    dec_alpha = alpha
    dec_alpha_pol = alpha_pol
    final_temp = 1e-5 # Final temperature
    temp = temperature # Current temperature
    convergence_t = 0
    
    # Curriculum parameters
    alpha_star = dec_alpha
    tau_star = 1
  
    # Mid-term results
    Qs = []
    thetas = []

    e = np.zeros_like(Q)
    l = lam
    for episode in range(episodes): # Each episode is a single time step
        
        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = select_action(tmdp.env.s, theta, temperature=temp)
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            flags["terminated"] = terminated
            sample = {"state": s, "action": a, "reward": r, "next_state": s_prime, "flags": flags, "t": t, "k": k}
            traj.append(sample)
            # Setup next step
            tmdp.env.s = s_prime
            k += 1
            t += 1
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
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
            
            compute_advantages(batch, V, l, tmdp.gamma)# Compute advantages during trajectories
            pi_old = get_softmax_policy(theta, temperature=temp)
            theta_ref = np.copy(theta)
            # Iterate over trajectories in the batch
            for _ in range(epochs):
                for trajectory in batch:
                    # Iterate over samples in the trajectory
                    for j, sample in enumerate(trajectory):
                        s = sample["state"]
                        a = sample["action"]
                        r = sample["reward"]
                        s_prime = sample["next_state"]
                        flags = sample["flags"]
                        t = sample["t"]
                        adv = sample["adv"]

                        # learning parameters decay
                        if tmdp.tau != 0:
                            temp = temperature 
                            dec_alpha= alpha
                            dec_alpha_pol = alpha_pol
                        else:
                            temp = temperature + (final_temp - temperature)*((episode - convergence_t)/(episodes-convergence_t))
                            dec_alpha= max(1e-5, alpha*(1 - (episode-convergence_t)/(episodes-convergence_t)))
                            dec_alpha_pol = max(1e-5, alpha_pol*(1 - (episode-convergence_t)/(episodes-convergence_t)))

                        pi = get_softmax_policy(theta_ref, temperature=temp)
                        ratio = pi[s,a]/pi_old[s,a]
                        # Compute the clipped surrogate objective
                        l_clip = np.minimum(ratio*adv, np.clip(ratio, 1-eps, 1+eps)*adv)
                        # Update the policy parameters
                        theta_ref[s][a] += dec_alpha_pol*l_clip*adv # implicit gradient calculation
                        # Evaluation step
                        td_error = dec_alpha*(r + tmdp.gamma*V[s_prime] - V[s])
                        V[s] = V[s] + td_error
                        U[s,a,s_prime] = U[s,a,s_prime] + dec_alpha*(r + tmdp.gamma*V[s_prime] - U[s,a,s_prime])
                
            # Reset the batch
            batch = []
            t = 0

            # Bound evaluation
            if( tmdp.tau > 0):
                l = 0.
                rel_pol_adv = compute_relative_policy_advantage_function(pi, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)
                delta_Q = get_sup_difference_Q(Q)

                delta_U = get_sup_difference_U(U)
                if delta_U == 0:
                    delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                    
                d_inf_pol = get_d_inf_policy(pi, pi_old)
                d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
                d_exp_pol = get_d_exp_policy(pi, pi_old, d)
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
                
                alpha_star, tau_star = get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds)

                print(optimal_pairs)
                print(teleport_bounds)
                print("Updating the policy with alpha_star: ", alpha_star, "tau_star: ",tau_star)
                tmdp.update_tau(tau_star)
                theta = alpha_star*theta_ref + (1-alpha_star)*theta
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    # Set lambda parameter for eligibility traces for the fine tuning of the policy
                    l=lam
                    print(Q)
                    print(theta)
                    conv_Q = np.copy(Q)
                    conv_theta = np.copy(theta)
            else:
                theta = theta_ref
                print("Working on the original problem, episode {}".format(episode))

        if episode % status_step == 0 or done:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(np.copy(Q))
            thetas.append(np.copy(theta))

    return {"Qs": Qs, "history": history, "thetas": thetas}#, "conv_Q": conv_Q, "conv_theta": conv_theta, "convergence_t": convergence_t}


def curriculum_PPO_1(tmdp:TMDP, Q, episodes=5000, epochs=10, alpha=.25, alpha_pol=.125, status_step=50000, batch_nS=1, temperature=1.0, lam=0., eps=0.2, biased=True):
    
    nS, nA = Q.shape
    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []
    # Subset of the history, considering only the current batch
    batch = []
    # Subset of the batch, considering only the current trajectory
    traj = []

    t = 0 # batch counter
    k = 0 # trajectory counter
    done = False 
    terminated = False

    #Policy parameter vector, considering nA parameters for each state
    # Random initialization
    theta = np.zeros((nS, nA))
    #theta = np.zeros((nS, nA))
    theta_ref = np.zeros((nS, nA))

    # State action next-state value function
    U = np.zeros((nS, nA, nS))

    # Hyperparameters decay
    dec_alpha = alpha
    dec_alpha_pol = alpha_pol
    final_temp = 1e-5 # Final temperature
    temp = temperature # Current temperature
    convergence_t = 0
    
    # Curriculum parameters
    alpha_star = dec_alpha
    tau_star = 1
  
    # Mid-term results
    Qs = []
    thetas = []

    e = np.zeros_like(Q)
    l = lam
    for episode in range(episodes): # Each episode is a single time step
        
        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = select_action(tmdp.env.s, theta, temperature=temp)
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            flags["terminated"] = terminated
            sample = {"state": s, "action": a, "reward": r, "next_state": s_prime, "flags": flags, "t": t, "k": k}
            traj.append(sample)
            # Setup next step
            tmdp.env.s = s_prime
            k += 1
            t += 1
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
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
            
            pi_old = get_softmax_policy(theta, temperature=temp)
            theta_ref = np.copy(theta)
            # Iterate over trajectories in the batch
            for _ in range(epochs):
                for trajectory in batch:
                    # Iterate over samples in the trajectory
                    for j, sample in enumerate(trajectory):
                        s = sample["state"]
                        a = sample["action"]
                        r = sample["reward"]
                        s_prime = sample["next_state"]
                        flags = sample["flags"]
                        t = sample["t"]
                        
                        # learning parameters decay
                        if tmdp.tau != 0:
                            temp = temperature 
                            dec_alpha= alpha
                            dec_alpha_pol = alpha_pol
                        else:
                            temp = temperature + (final_temp - temperature)*((episode - convergence_t)/(episodes-convergence_t))
                            dec_alpha= max(1e-5, alpha*(1 - (episode-convergence_t)/(episodes-convergence_t)))
                            dec_alpha_pol = max(1e-5, alpha_pol*(1 - (episode-convergence_t)/(episodes-convergence_t)))
                        
                        # Picking next action
                        if flags["done"] or flags["terminated"]:
                            a_prime = select_action(tmdp.env.s, theta, temperature=temp) # Last element of trajectory, pick the action from the policy
                        else:
                            a_prime = trajectory[j+1]["action"] # Pick next action from next state
                        

                        #a_prime = greedy(s_prime, Q, tmdp.env.allowed_actions[int(s_prime)])
                        
                        td_error = dec_alpha*(r + tmdp.gamma*Q[s_prime, a_prime] - Q[s,a])
                        e[s,a] = 1
                        if l == 0:
                            Q[s,a] = Q[s,a] + e[s,a]*td_error
                        else:
                            for s_1 in range(nS):
                                for a_1 in range(nA):
                                    Q[s_1,a_1] = Q[s_1,a_1] + e[s_1,a_1]*td_error
                        e = tmdp.gamma*l*e

                        # Get current policy
                        pi = get_softmax_policy(theta_ref, temperature=temp)
                        V = compute_V_from_Q(Q, pi)
                        adv = Q[s,a] - V[s]

                        U[s,a,s_prime] = U[s,a,s_prime] + dec_alpha*(r + tmdp.gamma*V[s_prime] - U[s,a,s_prime])
                        ratio = pi[s,a]/pi_old[s,a]
                        # Compute the clipped surrogate objective
                        l_clip = np.minimum(ratio*adv, np.clip(ratio, 1-eps, 1+eps)*adv)
                        # Update the policy parameters
                        theta_ref[s][a] += dec_alpha_pol*l_clip*adv # implicit gradient calculation
                        pi = get_softmax_policy(theta_ref, temperature=temp)
                
            # Reset the batch
            batch = []
            t = 0

            # Bound evaluation
            if( tmdp.tau > 0):
                l = 0.
                rel_pol_adv = compute_relative_policy_advantage_function(pi, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)
                delta_Q = get_sup_difference_Q(Q)

                delta_U = get_sup_difference_U(U)
                if delta_U == 0:
                    delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                    
                d_inf_pol = get_d_inf_policy(pi, pi_old)
                d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
                d_exp_pol = get_d_exp_policy(pi, pi_old, d)
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
                
                alpha_star, tau_star = get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds)

                print(optimal_pairs)
                print(teleport_bounds)
                print("Updating the policy with alpha_star: ", alpha_star, "tau_star: ",tau_star)
                tmdp.update_tau(tau_star)
                theta = alpha_star*theta_ref + (1-alpha_star)*theta
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    # Set lambda parameter for eligibility traces for the fine tuning of the policy
                    l=lam
                    print(Q)
                    print(theta)
            else:
                theta = theta_ref
                print("Working on the original problem, episode {}".format(episode))

        if episode % status_step == 0 or done:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(np.copy(Q))
            thetas.append(np.copy(theta))

    return {"Qs": Qs, "history": history, "thetas": thetas}#, "conv_Q": conv_Q, "conv_theta": conv_theta, "convergence_t": convergence_t}




"""
    Double Q_learning algorithm implementation. It tries to learn Q_tau as a mixture among Q_p and Q_xi, where the mixture is controlled by the teleportation probability tau.
    Args:
        - tmdp (TMDP): environment object
        - Q_p (ndarray): state-action value function for the original problem
        - Q_xi (ndarray): state-action value function for the simplified problem
        - episodes (int): number of episodes to be done
        - alpha (float): initial learning rate
        - status_step (int): intermediate results flag. Used for the evaluation of the state action value function updates while learning
        - batch_nS (int): nS of the batch
    return (dict): list of state action value functions updated at each status step, visit distributions, visit count, history of episodes
"""
def batch_double_q_learning(tmdp:TMDP, Q_p, Q_xi, episodes=5000, alpha=1., eps=0., status_step=5000, batch_nS=1):

    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []

    # Subset of the history, considering only the current batch
    batch = []
    trajectory_count = 0
    t = 0
    done = False

    nS, nA = Q_p.shape

    # Count the number of visits to each state
    visits = np.zeros(nS)
    visits_distr = np.zeros(nS)
    disc_visits = np.zeros(nS)
    disc_visits_distr = np.zeros(nS)
   
    if not eps:
        eps = min(1, alpha*2)
    
    # List of state action value functions updated at each status step
    Qs = []
    Q_ps = []
    Q_xis = []

    Q = (1-tmdp.tau)*Q_p + tmdp.tau*Q_xi

    visits_distributions = []
    disc_visits_distributions = []
    
    dec_alpha = alpha
    dec_eps = eps
    avg_reward = 0

    for episode in range(episodes):
        # Q_learning main loop
        cum_return = 0
        while True:
            s = tmdp.env.s
            # Pick an action according to the epsilon greedy policy
            a = eps_greedy(tmdp.env.s, Q, dec_eps, tmdp.env.allowed_actions[int(tmdp.s)])
          
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            his = {"state": s, "action": a, "reward": r, "next_state": s_prime, "flags": flags, "t": t}
            history.append(his)
            batch.append(his)
            
            # Setup next step
            tmdp.env.s = s_prime

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

                cum_return += r*tmdp.gamma**ep_count

                # Policy improvement step, greedy w.r.t. Q
                a_prime = greedy(s_prime, Q, tmdp.env.allowed_actions[int(s_prime)])

                # Evaluation step
                if not flags["teleport"]: # Update Q_p
                    Q_p[s,a] = Q_p[s,a] + dec_alpha*(r + tmdp.gamma*Q[s_prime, a_prime] - Q_p[s,a])
                    #Q_p[s,a] = Q_p[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s_prime, a_prime]+ tmdp.tau*Q_xi[s_prime, a_prime]) - Q_p[s,a])
                else: # Update Q_xi
                    Q_xi[s,a] = Q_xi[s,a] + dec_alpha*(r + tmdp.gamma*Q[s_prime, a_prime] - Q_xi[s,a])
                    #Q_xi[s,a] = Q_xi[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s_prime, a_prime]+ tmdp.tau*Q_xi[s_prime, a_prime]) - Q_xi[s,a])
                
                # Update Q 
                # Option 1 - Leaning Q from Q_p and Q_xi
                #Q[s,a] = Q[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s_prime, a_prime]+ tmdp.tau*Q_xi[s_prime, a_prime]) - Q[s,a])

                # Option 2 - Leaning Q from Q_p and Q_xi
                #Q[s,a] = Q[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s, a]+ tmdp.tau*Q_xi[s, a]) - Q[s,a])

                # Option 3 - Compute Q directly from Q_p and Q_xi
                Q[s,a] = (1-tmdp.tau)*Q_p[s, a]+ tmdp.tau*Q_xi[s, a]

                visits[s] += 1
                disc_visits[s] += tmdp.gamma**ep_count

                visits_distr = visits/(np.sum(visits))
                disc_visits_distr = disc_visits/(np.sum(disc_visits))

                if flags["done"]:
                    ep_count = 0
                else:
                    ep_count += 1

            avg_reward = cum_return/trajectory_count 

            # Reset the batch
            batch = []
            trajectory_count = 0

        if episode % status_step == 0 or done:
            # Mid-result status update
            Qs.append(Q.copy())
            Q_ps.append(Q_p.copy())
            Q_xis.append(Q_xi.copy())
            visits_distributions.append(visits_distr.copy())
            disc_visits_distributions.append(disc_visits_distr.copy())

    return {"Qs": Qs, "Q_ps": Q_ps, "Q_xis": Q_xis, "visits_distributions":visits_distributions, "visits": visits, "history": history, "disc_visits_distributions": disc_visits_distributions}

"""
    Policy gradient algorithm implementation
    Args:
        - tmdp (DiscreteEnv): environment object
        - Q (ndarray): inizial state-action value function
        - episodes (int): number of episodes to be done
        - alpha (float): initial learning rate
        - batch_nS (int): nS of the batch in terms of number of trajectories
        - status_step (int): intermediate results flag. Used for the evaluation of the state action value function updates while learning
    return (dict): list of state action value functions updated at each status step, history of episodes
"""
def policy_gradient(tmdp:TMDP, Q_p, Q_xi, episodes=5000, alpha=1., status_step=5000, batch_nS=1, temperature=1.0):

    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []
    # Subset of the history, considering only the current batch
    batch = []
    nS, nA = Q_p.shape

    Q = (1-tmdp.tau)*Q_p + tmdp.tau*Q_xi

    # Counts the number of trajectories in the current batch
    trajectory_count = 0
    t = 0 # total number of time steps. At the end t = episodes + #timesteps_last_trajectory (number of extra time steps to conclude the last trajectory)
    done = False 

    #Policy parameter vector, considering nA-1 parameters for each state
    theta = np.zeros((nS, nA-1))
    
    # Hyperparameters decay
    dec_alpha = alpha
    final_temp = 1e-5 # Final temperature
    temp = temperature # Current temperature
    
    # Visit distribution estimation
    visits = np.zeros(nS)
    visits_distr = np.zeros(nS)
    disc_visits = np.zeros(nS)
    disc_visits_distr = np.zeros(nS)
    
    # Mid-term results
    Qs = []
    Q_ps = []
    Q_xis = []
    thetas = []
    visits_distributions = []
    disc_visits_distributions = []

    for episode in range(episodes): # Each episode is a single time step
        
        # Sampling episodes from trajectories
        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = select_action(tmdp.env.s, theta, temperature=temp)
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            his = {"state": s, "action": a, "reward": r, "next_state": s_prime, "flags": flags, "t": t}
            history.append(his)
            batch.append(his)
            # Setup next step
            tmdp.env.s = s_prime

            t += 1
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
                tmdp.reset()
                trajectory_count += 1
            
            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                if flags["done"]:
                    done = True
                    break
        
        # Gradient policy over trajectories
        if( (trajectory_count != 0 and trajectory_count % batch_nS == 0) or done):
            
            # Estimation of the number of batches per episode if all batches has the nS of the current one
            batch_per_episode = np.ceil(episode/len(batch))
            # Estimation of the total number of batches if all batches has the nS of the current one
            total_number_of_batches = np.ceil(episodes/len(batch))
            
            # Temperature decay
            temp = temperature + (final_temp - temperature)*(batch_per_episode/total_number_of_batches)

            dec_alpha= max(0, alpha*(1-batch_per_episode/total_number_of_batches))
            ep_count = 0
            for j, ep in enumerate(batch):
                s = ep["state"]
                a = ep["action"]
                r = ep["reward"]
                s_prime = ep["next_state"]
                flags = ep["flags"]
                t = ep["t"]

                # Visit distribution update
                visits[s] += 1
                disc_visits[s] += tmdp.gamma**ep_count
                visits_distr = visits/(np.sum(visits))
                disc_visits_distr = disc_visits/(np.sum(disc_visits))   
                
                # Picking next action
                if flags["done"]:
                    a_prime = select_action(tmdp.env.s, theta, temperature=temp) # Last element of trajectory, pick the action from the policy
                    ep_count = 0 # Useful for batch_nS > 1
                else:
                    a_prime = batch[j+1]["action"] # Pick next action from next state
                    ep_count += 1 # Increase the time step within the trajectory
                
               # Evaluation step
                if not flags["teleport"]: # Update Q_p
                    Q_p[s,a] = Q_p[s,a] + dec_alpha*(r + tmdp.gamma*Q[s_prime, a_prime] - Q_p[s,a])
                    #Q_p[s,a] = Q_p[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s_prime, a_prime]+ tmdp.tau*Q_xi[s_prime, a_prime]) - Q_p[s,a])
                else: # Update Q_xi
                    Q_xi[s,a] = Q_xi[s,a] + dec_alpha*(r + tmdp.gamma*Q[s_prime, a_prime] - Q_xi[s,a])
                    #Q_xi[s,a] = Q_xi[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s_prime, a_prime]+ tmdp.tau*Q_xi[s_prime, a_prime]) - Q_xi[s,a])
                
                # Update Q 
                # Option 1 - Learning Q from Q_p and Q_xi
                #Q[s,a] = Q[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s_prime, a_prime]+ tmdp.tau*Q_xi[s_prime, a_prime]) - Q[s,a])

                # Option 2 - Learning Q from Q_p and Q_xi
                #Q[s,a] = Q[s,a] + dec_alpha*(r + tmdp.gamma*((1-tmdp.tau)*Q_p[s, a]+ tmdp.tau*Q_xi[s, a]) - Q[s,a])

                # Option 3 - Compute Q directly from Q_p and Q_xi
                Q[s,a] = (1-tmdp.tau)*Q_p[s, a]+ tmdp.tau*Q_xi[s, a]

                # Policy Gradient
                probabilities = softmax_policy(theta[s], temperature=temp)
                # Computing the gradient of the log policy
                grad_log_policy = -probabilities

                # For the taken action, a +1 is considerd if it is not the explicit one
                if a < nA-1:
                    grad_log_policy[a] += 1
                else:
                    # gradient set to 0 for the explicit action
                    grad_log_policy[a] = 0

                grad_log_policy = grad_log_policy/temp
                    
                theta[s] += dec_alpha*grad_log_policy[:-1]*Q[s,a] # Policy parameters update

            # Reset the batch
            batch = []
            trajectory_count = 0

        if episode % status_step == 0 or done:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(Q.copy())
            thetas.append(theta.copy())
            Q_ps.append(Q_p.copy())
            Q_xis.append(Q_xi.copy())
            visits_distributions.append(visits_distr.copy())
            disc_visits_distributions.append(disc_visits_distr.copy())

    return {"Qs": Qs, "Q_ps": Q_ps, "Q_xis": Q_xis, "visits_distributions":visits_distributions, "visits": visits, "history": history, "disc_visits_distributions": disc_visits_distributions, "thetas": thetas}

def curriculum_AC(tmdp:TMDP, Q, episodes=5000, alpha=.25, alpha_pol=.1, status_step=50000, 
                  batch_nS=1, temperature=1.0, lam=0., biased=True, epochs=1, use_delta_Q=False,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), final_temperature=1e-3):
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

    decay = 1
    temp_decay = 1
    grad_log_policy = np.zeros(nA)

    # Tensor conversion
    tensor_mu = torch.tensor(tmdp.env.mu, dtype=torch.float32).to(device)
    tensor_P_mat = torch.tensor(tmdp.env.P_mat, dtype=torch.float32).to(device)
    tensor_xi = torch.tensor(tmdp.xi, dtype=torch.float32).to(device)
    stucked_count = 0
    episode = 0

    d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
    while episode < episodes: # Each episode is a single time step

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
                k += 1
                t += 1
                sample = (s, a, r, s_prime, a_prime, flags, t, k)
                traj.append(sample)
                rewards.append(r)
                
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
            episode += 1

            if episode == episodes-1 and alpha_star > 0 and tmdp.tau == 0 and stucked_count == 0:
                episodes += 100000

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                flags["terminated"] = terminated
                if len(traj) > 0:
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
                    e = np.zeros((nS, nA))
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
                        V = compute_V_from_Q(Q, pi_ref)
                        U[s,a,s_prime] += alpha*decay*(r + tmdp.gamma*V[s_prime] - U[s,a,s_prime])

                        # Policy Gradient
                        grad_log_policy = -pi_ref[s]
                        grad_log_policy[a] += 1 # Sum 1 for the taken action
                        grad_log_policy = grad_log_policy/(temperature*temp_decay)
                        theta_ref[s] += alpha_pol*decay*grad_log_policy*(Q[s,a] - V[s]) # Policy parameters update
                        ep += 1 # Increase the episode counter

                        """print("Q[{}][{}] = {}, V[{}] = {}, U[{}][{}][{}] = {}".format(s,a,Q[s,a],s,V[s],s,a,s_prime,U[s,a,s_prime]))
                        print("Policy[{}] = {}, Advantage[{},{}] = {}".format(s,pi_ref[s],s, a,Q[s,a] - V[s]))
                        print("Policy gradient[{}]= {}".format(s, grad_log_policy))"""
            
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
                stucked_count = 0

            if alpha_star != 0:
                theta = alpha_star*theta_ref + (1-alpha_star)*theta
                stucked_count = 0


            decay = max(1e-8, 1-(ep)/(episodes))
            temp_decay = temperature + (final_temperature - temperature)*(ep/episodes)

            # Reset the batch
            batch = []
            reward_records.append(r_sum)
            rewards = []
            t = 0
            teleport_count = 0
            e_time = time.time()
            print("Time for bound evaluation: ", e_time - s_time)
            if pol_adv < 0 and tmdp.tau == 0:
                print("No further improvement possible. Episode: {}".format(episode))
                stucked_count += 1
                episode = min(episodes-1, episode+10000)
                ep = min(episodes-1, ep+10000)
                if stucked_count > 10:
                    Qs.append(np.copy(Q))
                    thetas.append(np.copy(theta))
                    break
        if episode % status_step == 0 or done or terminated:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(np.copy(Q))
            thetas.append(np.copy(theta))

    return {"Qs": Qs, "history": history, "thetas": thetas, "reward_records": reward_records}


def curriculum_Q_learning(tmdp:TMDP, Q, curriculum:list=[], episodes:int=5000, alpha:float=.25, eps:float=.4, status_step:int=50000, 
                  batch_nS:int=1, lam:float=0.):
    nS, nA = Q.shape

    if len(curriculum) == 0:
        curriculum = [1.0 - i*0.05 for i in range(21)]
    curr_index = min(1, len(curriculum)-1)
    tmdp.update_tau(curriculum[curr_index])
    
    # History as list of dictionaries {state, action, reward, next_state, flags, t} over all transactions
    history = []
    # Subset of the history, considering only the current batch
    batch = []
    # Subset of the batch, considering only the current trajectory
    traj = []

    rewards = []
    reward_records = []
    exp_performance = []

    t = 0 # episode in batch counter
    k = 0 # episode in trajectory counter

    done = False 
    terminated = False

    # Mid-term results
    Qs = []

    decay = 1
    eps_decay = 1
    episode = 0
    teleport_count = 0

    while episode < episodes: # Each episode is a single time step

        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = eps_greedy(tmdp.env.s, Q, eps*eps_decay, tmdp.env.allowed_actions[int(tmdp.env.s)])
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            
            flags["terminated"] = terminated
            
            if not flags["teleport"]:
                k += 1
                t += 1
                sample = (s, a, r, s_prime, flags, t, k)
                traj.append(sample)
                rewards.append(r)
                
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
            episode += 1

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                flags["terminated"] = terminated
                if len(traj) > 0:
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
            for trajectory in batch:
                # Iterate over samples in the trajectory
                e = np.zeros((nS, nA))
                for j, sample in enumerate(trajectory):
                    s, a, r, s_prime, flags, t, k  = sample
                    
                    a_prime = greedy(s_prime, Q, tmdp.env.allowed_actions[int(s_prime)]) 

                    td_error = alpha*decay*(r + tmdp.gamma*Q[s_prime, a_prime] - Q[s,a])

                    # Eligibility traces
                    e[s,a] = 1 # Freequency heuristic with saturation
                    if lam == 0:
                        Q[s,a] += e[s,a]*td_error
                    else:
                        Q = Q + e*td_error
                    e = tmdp.gamma*lam*e # Recency heuristic decay
            
            r_sum = sum(rewards)

            decay = max(1e-8, 1-(episode)/(episodes))
            eps_decay = max(0, (1-episode/episodes)**2)
            # Reset the batch
            batch = []
            reward_records.append(r_sum)
            V = compute_V_from_Q(Q, get_policy(Q))
            exp_performance.append(compute_expected_j(V, tmdp.env.mu))
            rewards = []
            t = 0
            teleport_count = 0

        # Curriculum update
        if episode > episodes/len(curriculum)*(curr_index) and tmdp.tau != 0:
            curr_index = min(curr_index+1, len(curriculum)-1)
            tmdp.update_tau(curriculum[curr_index])
            print("Curriculum update, episode:", episode, "tau:", curriculum[curr_index])

        if episode % status_step == 0 or done or terminated:
            #print("Mid-result status update, episode:", episode, "done:", done)
            # Mid-result status update
            Qs.append(np.copy(Q))

    return {"Qs": Qs, "history": history, "reward_records": reward_records, "exp_performance": exp_performance}


def curriculum_AC_buffer(tmdp:TMDP, Q, rep_buffer:ReplayBuffer, episodes=5000,  alpha=.25, alpha_pol=.1, status_step=50000, 
                  batch_nS=1, temperature=1.0, lam=0., biased=True, epochs=1, use_delta_Q=False,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), final_temperature=1e-3):
    nS, nA = Q.shape

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

    decay = 1
    temp_decay = 1
    grad_log_policy = np.zeros(nA)

    # Tensor conversion
    tensor_mu = torch.tensor(tmdp.env.mu, dtype=torch.float32).to(device)
    tensor_P_mat = torch.tensor(tmdp.env.P_mat, dtype=torch.float32).to(device)
    tensor_xi = torch.tensor(tmdp.xi, dtype=torch.float32).to(device)
    stucked_count = 0
    episode = 0

    d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
    while episode < episodes: # Each episode is a single time step

        pi = get_softmax_policy(theta, temperature=temperature*temp_decay)
        while True:
            s = tmdp.env.s
            # Pick an action according to the parametric policy
            a = select_action(pi[s])
            # Perform a step in the environment, picking action a
            s_prime, r, flags, p =  tmdp.step(a)
            a_prime = select_action(pi[s_prime])
            flags["terminated"] = terminated
            sample = (s, a, r, s_prime, a_prime, flags['done'])

            if not flags["teleport"]:
                rep_buffer.store_transition(*sample)
                rewards.append(r)
                k += 1
                t += 1
                if flags["done"]:
                    tmdp.reset()
                    k = 0 # Reset the episode counter
            else:
                teleport_count += 1 # Increase the teleportation counter

                if k > 0: # Not empty trajectory, termiated by teleportation
                    rep_buffer.end_trajectory()
                    k = 0 # Reset the episode counter
            
            episode += 1

            if episode == episodes-1 and alpha_star > 1 and tmdp.tau == 0 and stucked_count == 0:
                episodes += 100000 

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                flags["terminated"] = terminated

                if k > 0: # Not empty trajectory, termiated by teleportation
                    rep_buffer.end_trajectory()
                    k = 0 # Reset the episode counter
                break # temporary ending condition To be Removed
                if flags["done"]:
                    done = True
                    break

        # Processing the batch
        if (rep_buffer.traj_counter % batch_nS == 0) or done or terminated:
            

            # Iterate over trajectories in the batch
            for _ in range(epochs):
                for trajectory in batch:
                    # Iterate over samples in the trajectory
                    e = np.zeros((nS, nA))
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
                        V = compute_V_from_Q(Q, pi_ref)
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
                stucked_count = 0

            if alpha_star != 0:
                theta = alpha_star*theta_ref + (1-alpha_star)*theta
                stucked_count = 0


            decay = max(1e-8, 1-(ep)/(episodes))
            temp_decay = temperature + (final_temperature - temperature)*(ep/episodes)

            # Reset the batch
            batch = []
            reward_records.append(r_sum)
            rewards = []
            t = 0
            teleport_count = 0
            e_time = time.time()
            print("Time for bound evaluation: ", e_time - s_time)
            if pol_adv < 0 and tmdp.tau == 0:
                print("No further improvement possible. Episode: {}".format(episode))
                stucked_count += 1
                episode = min(episodes-1, episode+10000)
                ep = min(episodes-1, ep+10000)
                if stucked_count > 10:
                    Qs.append(np.copy(Q))
                    thetas.append(np.copy(theta))
                    break
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



def is_prime(n):
    """Check if a number is a prime number."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def generate_primes(start, count, step):
    """Generate a list of prime numbers starting from a given number."""
    primes = []
    num = start
    while len(primes) < count:
        if is_prime(num):
            count +=1
            if count % step == 0:
                primes.append(num)
        num += 1
    return primes

# Generate 40 prime numbers greater than 40
primes = generate_primes(1000, 30, 30)
print(primes)