import numpy as np
from gymnasium import Env
from DiscreteEnv import DiscreteEnv
from TMDP import TMDP
from model_functions import *
from gymnasium.utils import seeding
import torch
import torch.nn as nn
from torch.nn import functional as F
from PolicyPi import PolicyPi
from ActorCritic import *


def curriculum_PG(tmdp:TMDP, policy_pi:PolicyPi, opt, episodes=5000, alpha=.25, batch_nS=1, biased=True):
    
    nS, nA = tmdp.env.nS, tmdp.env.nA
    done = False 
    terminated = False

    batch_size = 0
    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    V = np.zeros(nS)
    Q = np.zeros((nS, nA))

    # Hyperparameters decay
    dec_alpha = alpha
    
    # Curriculum parameters
    alpha_star = dec_alpha
    tau_star = 1
    convergence_t = 0

    # Mid-term results
    reward_records = []
    states = []
    actions = []
    rewards = []
    flags_list = []
    t = 0
    for episode in range(episodes): # Each episode is a single time step
        
        while True:
            s = tmdp.env.s
            states.append(s)
            a = policy_pi.act(s) # Select action from policy

            s_prime, r, flags, p =  tmdp.step(a)

            flags["terminated"] = terminated
            actions.append(a)
            rewards.append(r)
            flags_list.append(flags)

            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
                tmdp.reset()
                batch_size += 1

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                flags["terminated"] = terminated
                break # temporary ending condition To be Removed
                if flags["done"]:
                    done = True
                    break

        # Processing the batch
        if( (batch_size != 0 and batch_size % batch_nS == 0) or done or terminated):
            # Extract previous policy for future comparison
            
            cum_reward = np.zeros_like(rewards)
            reward_len = len(rewards)
            for i in reversed(range(reward_len)):
                cum_reward[i] = rewards[i] + tmdp.gamma*cum_reward[i+1] if i < reward_len-1 else rewards[i]
                
            logits_old = policy_pi.get_logits()
            pi_old = policy_pi.get_probabilities()

            tensor_states = torch.tensor(states, dtype=torch.long).to(policy_pi.device)
            tensor_actions = torch.tensor(actions, dtype=torch.long).to(policy_pi.device)
            tensor_cum_reward = torch.tensor(cum_reward, dtype=torch.float).to(policy_pi.device)
            opt.zero_grad()
            logits = policy_pi(tensor_states)
            # Calculate negative log probability (-log P) as loss.
            # Cross-entropy loss is -log P in categorical distribution.

            logs_prob = -F.cross_entropy(logits, tensor_actions, reduction='none')
            loss = -logs_prob * tensor_cum_reward
            loss.sum().backward()
            opt.step()

            # Learning the value function
            for i in range(len(states)):
                t += 1
                dec_alpha= max(1e-5, alpha*(1 - t/episodes))
                s = states[i]
                a = actions[i]
                s_prime = states[i+1] if i < len(states)-1 else states[i]
                a_prime = actions[i+1] if i < len(states)-1 else pi_old[s].argmax()
                r = rewards[i]
                flags = flags_list[i]

                td_error = dec_alpha*(r + tmdp.gamma*V[s_prime] - V[s])
                V[s] = V[s] + dec_alpha*td_error
                U[s,a,s_prime] = U[s,a,s_prime] + dec_alpha*(r + tmdp.gamma*V[s_prime] - U[s,a,s_prime])
                Q[s,a] = Q[s,a] + dec_alpha*(r + tmdp.gamma*Q[s_prime, a_prime] - Q[s,a])

            # Bound evaluation
            if( tmdp.tau > 0):
                pi = policy_pi.get_probabilities()
                rel_pol_adv = compute_relative_policy_advantage_function(pi, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)

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
    
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    # Set lambda parameter for eligibility traces for the fine tuning of the policy
            r_sum = sum(rewards)
            print("Running episode {} reward {}".format(episode, r_sum))
            # Reset the batch
            states = []
            actions = []
            rewards = []
            flags_list = []
            batch_size = 0
            reward_records.append(r_sum)

    return {"Q": Q, "V": V, "U": U, "policy_pi": policy_pi, "convergence_t": convergence_t, "reward_records": reward_records}


def curriculum_ActorCritic(tmdp:TMDP, policy_pi:ActorNet, value_func:ValueNet, q_func:QNet, pol_opt, val_opt, q_opt, alpha=1., episodes=5000, batch_nS=1, biased=True):
    
    nS, nA = tmdp.env.nS, tmdp.env.nA
    done = False 
    terminated = False
    converged = False

    batch_size = 0
    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    
    # Curriculum parameters
    alpha_star = 1
    tau_star = 1
    convergence_t = 0

    # Mid-term results
    reward_records = []
    states = []
    actions = []
    rewards = []
    flags_list = []
    t = 0
    for episode in range(episodes): # Each episode is a single time step
        
        # Sample the environment
        while True:
            s = tmdp.env.s
            states.append(s)
            a = policy_pi.act(s) # Select action from policy

            s_prime, r, flags, p =  tmdp.step(a)

            flags["terminated"] = terminated
            actions.append(a)
            rewards.append(r)
            flags_list.append(flags)
            v_next = 0
            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
                tmdp.reset()
                batch_size += 1
            else:
                with torch.no_grad():
                    v_next = value_func.get_value(s_prime)
            
            dec_alpha = max(1e-8, alpha*(1 - episode/episodes))
            U[s,a,s_prime] += dec_alpha *(r + tmdp.gamma*v_next - U[s,a,s_prime])

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                flags["terminated"] = terminated
                break # temporary ending condition To be Removed
                if flags["done"]:
                    done = True
                    break

        # Processing the batch
        if( (batch_size != 0 and batch_size % batch_nS == 0) or done or terminated):
            # Extract previous policy for future comparison
            
            cum_reward = np.zeros_like(rewards)
            reward_len = len(rewards)
            for i in reversed(range(reward_len)):
                cum_reward[i] = rewards[i] + (tmdp.gamma*cum_reward[i+1] if i < reward_len-1 else rewards[i])

            # Critic Optimization
            val_opt.zero_grad()
            tensor_states = torch.tensor(states, dtype=torch.long).to(value_func.device)
            
            tensor_cum_reward = torch.tensor(cum_reward, dtype=torch.float).to(value_func.device)
            tensor_values = value_func(tensor_states).squeeze(dim=1)
            vf_loss = F.mse_loss(tensor_values, tensor_cum_reward, reduction='none')
            vf_loss.sum().backward()
            val_opt.step()
            
            # QNet Optimization
            q_opt.zero_grad()
            tensor_actions = torch.tensor(actions, dtype=torch.long).to(q_func.device)

            tensor_q_values = q_func(tensor_states, tensor_actions).squeeze(dim=1)
            tensor_done = torch.tensor([flags["done"] for flags in flags_list], dtype=torch.bool).to(q_func.device)

            q_loss = F.mse_loss(tensor_q_values, tensor_cum_reward, reduction='none')
            q_loss.sum().backward()
            q_opt.step()

            # Policy Optimization    
            with torch.no_grad():
                tensor_values = value_func(tensor_states) # Get value function's values after learning
            
            pol_opt.zero_grad()

            logits_old = policy_pi.get_logits()
            pi_old = policy_pi.get_probabilities()

            tensor_advantages = tensor_cum_reward - tensor_values # Compute advantages of visited states

            logits = policy_pi(tensor_states) # Get logits of visited states
            logs_prob = -F.cross_entropy(logits, tensor_actions, reduction='none') # Compute log probabilities of visited states
            pi_loss = -logs_prob * tensor_advantages
            pi_loss.sum().backward()
            pol_opt.step()

            # Bound evaluation
            if( tmdp.tau > 0):
                pi = policy_pi.get_probabilities()
                Q = q_func.get_values()
                rel_pol_adv = compute_relative_policy_advantage_function(pi, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)

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
    
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    # Set lambda parameter for eligibility traces for the fine tuning of the policy
            
            r_sum = sum(rewards)
            print("Running episode {} reward {}".format(episode, r_sum))
            # Reset the batch
            states = []
            actions = []
            rewards = []
            flags_list = []
            batch_size = 0
            reward_records.append(r_sum)

    return {"policy_pi": policy_pi, "value_func":value_func, "convergence_t": convergence_t, "reward_records": reward_records}

def curriculum_PPO(tmdp:TMDP, policy_pi:ActorNet, value_func:ValueNet, q_func:QNet, loss_opt, q_opt, alpha=1., eps=.2, vf_coeff=0.5, h_coeff=0.1, episodes=5000, batch_nS=1, biased=True):
    
    nS, nA = tmdp.env.nS, tmdp.env.nA
    done = False 
    terminated = False
    converged = False

    batch_size = 0
    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    
    # Curriculum parameters
    alpha_star = 1
    tau_star = 1
    convergence_t = 0

    # Mid-term results
    reward_records = []
    states = []
    actions = []
    logit_list = []
    logprob_list = []
    next_states = []
    rewards = []
    flags_list = []
    dec_h_coeffs = []
    t = 0
    for episode in range(episodes): # Each episode is a single time step
        
        # Sample the environment
        while True:
            s = tmdp.env.s
            states.append(s)
            a, l, p = policy_pi.act_and_log_prob(s) # Select action from policy
            s_prime, r, flags, _ =  tmdp.step(a)
            #print("State: ", s, "Action: ", a, "Next State: ", s_prime, "Reward: ", r, "Done: ", flags["done"])
            flags["terminated"] = terminated

            actions.append(a)
            rewards.append(r)
            logit_list.append(l)
            logprob_list.append(p)
            next_states.append(s_prime)
            flags_list.append(flags)
            dec_h_coeffs.append(h_coeff*(1 - episode/episodes))
            v_next = 0

            # Reset the environment if a terminal state is reached or if a teleportation happened
            if flags["done"]:# or flags["teleport"]:
                tmdp.reset()
                batch_size += 1
                if r > 0:
                    print("Found a terminal state, cumulative reward: {}, episode {} tau ".format(sum(rewards), episode, tmdp.tau))
            else:
                with torch.no_grad():
                    v_next = value_func.get_value(s_prime)
            
            dec_alpha = max(1e-8, alpha*(1 - episode/episodes))
            U[s,a,s_prime] += dec_alpha *(r + tmdp.gamma*v_next - U[s,a,s_prime])

            if episode < episodes-1: # move to next time step
                break   
            else: # if reached the max num of time steps, wait for the end of the trajectory for consistency
                print("Ending the loop")
                terminated = True
                flags["terminated"] = terminated
                break # temporary ending condition To be Removed
                if flags["done"]:
                    done = True
                    break

        # Processing the batch
        if( (batch_size != 0 and batch_size % batch_nS == 0) or done or terminated):
            # Extract previous policy for future comparison
            
            cum_reward = np.zeros_like(rewards)
            reward_len = len(rewards)
            for i in reversed(range(reward_len)):
                cum_reward[i] = rewards[i] + (tmdp.gamma*cum_reward[i+1] if i < reward_len-1 else rewards[i])

            pi_old = policy_pi.get_probabilities()

            # Combined Loss Optimization
            loss_opt.zero_grad()

            tensor_states = torch.tensor(states, dtype=torch.long).to(value_func.device)
            tensor_actions = torch.tensor(actions, dtype=torch.long).to(q_func.device)
            tensor_next_states = torch.tensor(next_states, dtype=torch.long).to(q_func.device)

            #tensor_logits_old = torch.tensor(logit_list, dtype=torch.float).to(policy_pi.device)
            
            tensor_logs_prob = torch.tensor(logprob_list, dtype=torch.float).to(policy_pi.device).unsqueeze(dim=1)
            tensor_cum_reward = torch.tensor(cum_reward, dtype=torch.float).to(value_func.device).unsqueeze(dim=1)
            
            tensor_values = value_func(tensor_states)
            tensor_logits_new = policy_pi(tensor_states)
            tensor_advantages = tensor_cum_reward - tensor_values # Compute advantages of visited states

            tensor_logs_prob_new = -F.cross_entropy(tensor_logits_new, tensor_actions, reduction='none').unsqueeze(dim=1)
            ratio = torch.exp(tensor_logs_prob_new - tensor_logs_prob)


            l_clip = torch.min(ratio*tensor_advantages, torch.clamp(ratio, 1-eps, 1+eps)*tensor_advantages)
            
            # Compute probabilities and log probabilities of next state for entropy calculation
            tensor_logits_next_state = policy_pi(tensor_next_states)
            next_state_prob = F.softmax(tensor_logits_next_state, dim=-1)
            next_state_log_prob = torch.log(next_state_prob)
            tensor_h_coeffs = torch.tensor(dec_h_coeffs, dtype=torch.float).to(policy_pi.device).unsqueeze(dim=1)
            h = next_state_prob.unsqueeze(dim=1) @ next_state_log_prob.unsqueeze(dim=2)
            h = h.squeeze(dim=1)
            h = - h @ tensor_h_coeffs.unsqueeze(dim=2)
            vf_loss = F.mse_loss(tensor_values, tensor_cum_reward, reduction='none')
            loss = -l_clip + vf_loss * vf_coeff
            loss.sum().backward()
            loss_opt.step()

            # QNet Optimization
            q_opt.zero_grad()
            tensor_q_values = q_func(tensor_states, tensor_actions)
            tensor_q_values_next = q_func(tensor_next_states, tensor_actions)
            tensor_rewards = torch.tensor(rewards, dtype=torch.float).to(q_func.device).unsqueeze(dim=1)
            tensor_done = torch.tensor([flags["done"] for flags in flags_list], dtype=torch.float).to(q_func.device).unsqueeze(dim=1)
            exp_q = tensor_rewards + tmdp.gamma*tensor_q_values_next.squeeze(dim=1).unsqueeze(dim=0) @ (1-tensor_done)
            q_loss = F.mse_loss(exp_q, tensor_q_values, reduction='none')
            q_loss.sum().backward()
            q_opt.step()

            
            logits_old = policy_pi.get_logits()

            pi = policy_pi.get_probabilities()
            # Bound evaluation
            if( tmdp.tau > 0):
                Q = q_func.get_values()
                rel_pol_adv = compute_relative_policy_advantage_function(pi, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)

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
                if alpha_star <= 1e-3:
                    tmdp.update_tau(tau_star)
                    print("Updating tau with tau_star: ",tau_star)
    
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    # Set lambda parameter for eligibility traces for the fine tuning of the policy
            
            r_sum = sum(rewards)
            print("Running episode {} reward {}".format(episode, r_sum))
            # Reset the batch
            states = []
            actions = []
            rewards = []
            next_states = []
            logit_list = []
            logprob_list = []
            dec_h_coeffs = []
            flags_list = []
            batch_size = 0
            reward_records.append(r_sum)

            if np.linalg.norm(pi - pi_old, np.inf) < 1e-4 and episode > episodes//10:
                print("Converged to a stable policy, episode {}".format(episode))
                convergence_t = episode
                converged = True
                break

    return {"policy_pi": policy_pi, "value_func":value_func, "q_func":q_func, "convergence_t": convergence_t, "reward_records": reward_records}

