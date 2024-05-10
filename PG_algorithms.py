import numpy as np
from gymnasium import Env
from DiscreteEnv import DiscreteEnv
from TMDP import TMDP
from model_functions import *
from gymnasium.utils import seeding
import torch
import torch.nn as nn
from torch.nn import functional as F
from ActorCritic import *
from algorithms import *
from model_functions import *
from ReplayBuffer import ReplayBuffer
import os

def curriculum_SAC(tmdp:TMDP, policy_pi:ActorNet, ref_policy:ActorNet, q1_target:QNet, 
                   q2_target:QNet, q1_func:QNet, q2_func:QNet, ref_opt:torch.optim.Optimizer, 
                   q1_opt:torch.optim.Optimizer, q2_opt:torch.optim.Optimizer, rep_buffer:ReplayBuffer, 
                   alpha=1., alpha_u=.2, beta=.2, episodes=5000, batch_size=1, sample_steps=1,
                   update_rate=10, biased=False):
    
    nS, nA = tmdp.env.nS, tmdp.env.nA
    done_ = False 
    terminated = False

    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    
    # Curriculum parameters
    alpha_star = 1
    tau_star = 1
    convergence_t = 0
    teleport_count = 0

    # Mid-term results
    reward_records = []
    rewards = []
    t = 0
    dec_alpha = alpha
    ref_policy.soft_update(policy_pi, 1.) # Sincronize the policies

    for episode in range(episodes): # Each episode is a single time step
        
        # Sample the environment
        while True:
            s = tmdp.env.s
            a = ref_policy.act(s) # Select action from policy
            s_prime, r, flags, _ =  tmdp.step(a)
            
            if not flags["teleport"]:
                sample = (s, a, r, s_prime, flags["done"])
                rep_buffer.store_transition(*sample)
                rewards.append(r) # Store the reward for the current step
                v_next = 0
                if flags["done"]:
                    tmdp.reset()
                else:
                    with torch.no_grad():
                        tensor_s_prime = torch.tensor([s_prime], dtype=torch.long).to(q1_func.device)
                        probs = ref_policy.get_probs(tensor_s_prime)
                        v_1 = q1_target(tensor_s_prime) @ probs
                        v_2 = q2_target(tensor_s_prime) @ probs
                        v_next = min(v_1, v_2)
            
                dec_alpha_u = max(1e-8, alpha_u*(1 - episode/episodes))
                U[s,a,s_prime] += dec_alpha_u*(r + tmdp.gamma*v_next - U[s,a,s_prime])
            else:
                teleport_count += 1
            
            if rep_buffer.len() >= batch_size:
                t += 1

            if episode >= episodes-1:
                if flags["done"]:
                    done_ = True
                else:
                    terminated = True
            break
            
        
        if (rep_buffer.len() >= batch_size and t % sample_steps == 0)  or done_ or terminated:
            r_sum = sum(rewards)

            # Processing the batch
            states, actions, rewards, next_states, done = rep_buffer.sample_buffer(batch_size)
            states = torch.tensor(states, dtype=torch.long).to(ref_policy.device)
            actions = torch.tensor(actions, dtype=torch.long).to(ref_policy.device).unsqueeze(dim=1)
            next_states = torch.tensor(next_states, dtype=torch.long).to(ref_policy.device)
            rewards = torch.tensor(rewards, dtype=torch.float).to(ref_policy.device).unsqueeze(dim=1)
            done = torch.tensor(done).to(ref_policy.device, dtype=torch.int).unsqueeze(dim=1)

            # Compute target value for Q function
            with torch.no_grad():
                next_actions, next_log_probs = ref_policy.sample(next_states.squeeze(dim=-1))

                target_q1 = q1_target(next_states).squeeze(dim=1)
                target_q1 = target_q1.gather(1, next_actions)

                target_q2 = q2_target(next_states).squeeze(dim=1)
                target_q2 = target_q2.gather(1, next_actions)

                target_v = torch.min(target_q1, target_q2) - dec_alpha*next_log_probs # Eq. 6 in the paper (SAC)
                y = rewards + tmdp.gamma*(1-done)*target_v # Eq. 8 in the paper (SAC)
            
            # Q functions update
            
            q1_value = q1_func(states).squeeze(dim=1).gather(1, actions)
            q1_loss = F.mse_loss(q1_value, y)
            q1_opt.zero_grad()
            q1_loss.backward()
            print("Q1 Loss: ", q1_loss.item())
            q1_opt.step()

            q2_value = q2_func(states).squeeze(dim=1).gather(1, actions)
            q2_loss = F.mse_loss(q2_value, y)
            q2_opt.zero_grad()
            q2_loss.backward()
            print("Q2 Loss: ", q2_loss.item())
            q2_opt.step()
            
            # Actor Update
            current_actions, current_log_probs = ref_policy.sample(states.squeeze(dim=-1))
            q1_actor_value = q1_func(states).squeeze(dim=1).gather(1, current_actions)
            q2_actor_value = q2_func(states).squeeze(dim=1).gather(1, current_actions)

            q_actor_value = torch.min(q1_actor_value, q2_actor_value)

            actor_loss = (dec_alpha*current_log_probs - q_actor_value).mean()
            ref_opt.zero_grad()
            actor_loss.backward()
            print("Actor Loss: ", actor_loss.item())
            ref_opt.step()

            # Target function soft update
            q1_target.soft_update(q1_func, beta)
            q2_target.soft_update(q2_func, beta)
            
            
            pi_ref = ref_policy.get_probabilities()
            pi_old = policy_pi.get_probabilities()

            dec_alpha = max(1e-8, alpha*(1 - episode/episodes))

            # Bound evaluation
            if( tmdp.tau > 0 and t % update_rate == 0):
                Q = np.minimum(q1_func.get_values(), q2_func.get_values())

                rel_pol_adv = compute_relative_policy_advantage_function(pi_ref, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)

                delta_U = get_sup_difference_U(U)
                #delta_U = get_sup_difference_Q(Q) # Valid only if reward associated to teleport is null
                if delta_U == 0:
                    delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                

                d_inf_pol = get_d_inf_policy(pi_ref, pi_old)
                d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
                d_exp_pol = get_d_exp_policy(pi_ref, pi_old, d)
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

                #print(optimal_pairs)
                #print(teleport_bounds)

                policy_pi.soft_update(ref_policy, 1)
                tmdp.update_tau(tau_star)
                print("Alpha*: {} tau*: {} Episode: {} length: {} #teleports:{}".format(alpha_star, tau_star, episode, len(rewards),teleport_count))
                if r_sum > 0:
                    print("Got not null reward {}!".format(r_sum))
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    #policy_pi.soft_update(ref_policy, 1)
        
            else:
                policy_pi.soft_update(ref_policy, 1.)
                print("Episode: {} length: {}".format(episode, len(rewards)))
                if r_sum > 0:
                    print("Got not null reward {}!".format(r_sum))
            
            reward_records.append(r_sum)

            # Reset the batch
            rewards = []   
            teleport_count = 0

    return {"convergence_t": convergence_t, "reward_records": reward_records}


def curriculum_PPO_exp_replay(tmdp:TMDP, policy_pi:ActorNet, ref_policy:ActorNet, v_net:ValueNet, 
                   q_func:QNet, loss_opt:torch.optim.Optimizer, q_opt:torch.optim.Optimizer,
                   rep_buffer:ReplayBuffer, alpha=1., alpha_u=.2, beta=.2, epsilon=.2, episodes=5000,
                   batch_size=1, sample_steps=1, update_rate=1, ppo_epochs=1, biased=False):
    
    nS, nA = tmdp.env.nS, tmdp.env.nA
    done_ = False 
    terminated = False

    # State action next-state value function
    U = np.zeros((nS, nA, nS))
    
    # Curriculum parameters
    alpha_star = 1
    tau_star = 1
    convergence_t = 0
    teleport_count = 0

    # Mid-term results
    reward_records = []
    rewards = []
    t = 0
    dec_alpha = alpha
    ref_policy.soft_update(policy_pi, 1.) # Sincronize the policies

    for episode in range(episodes): # Each episode is a single time step
        
        # Sample the environment
        while True:
            s = tmdp.env.s
            a = policy_pi.act(s) # Select action from policy
            s_prime, r, flags, _ =  tmdp.step(a)
            
            if not flags["teleport"]:
                sample = (s, a, r, s_prime, flags["done"])
                rep_buffer.store_transition(*sample)
                rewards.append(r) # Store the reward for the current step
                v_next = 0
                if flags["done"]:
                    tmdp.reset()
                else:
                    with torch.no_grad():
                        tensor_s_prime = torch.tensor([s_prime], dtype=torch.long).to(q1_func.device)
                        probs = ref_policy.get_probs(tensor_s_prime)
                        v_1 = q1_target(tensor_s_prime) @ probs
                        v_2 = q2_target(tensor_s_prime) @ probs
                        v_next = min(v_1, v_2)
            
                dec_alpha_u = max(1e-8, alpha_u*(1 - episode/episodes))
                U[s,a,s_prime] += dec_alpha_u*(r + tmdp.gamma*v_next - U[s,a,s_prime])
            else:
                teleport_count += 1
            
            if rep_buffer.len() >= batch_size:
                t += 1

            if episode >= episodes-1:
                if flags["done"]:
                    done_ = True
                else:
                    terminated = True
            break
            
        
        if (rep_buffer.len() >= batch_size and t % sample_steps == 0)  or done_ or terminated:
            r_sum = sum(rewards)

            # Processing the batch
            states, actions, rewards, next_states, done = rep_buffer.sample_buffer(batch_size)
            states = torch.tensor(states, dtype=torch.long).to(ref_policy.device)
            actions = torch.tensor(actions, dtype=torch.long).to(ref_policy.device).unsqueeze(dim=1)
            next_states = torch.tensor(next_states, dtype=torch.long).to(ref_policy.device)
            rewards = torch.tensor(rewards, dtype=torch.float).to(ref_policy.device).unsqueeze(dim=1)
            done = torch.tensor(done).to(ref_policy.device, dtype=torch.int).unsqueeze(dim=1)
            
            for _ in range(ppo_epochs):
                # Compute the Advantage
                with torch.no_grad():
                    tensor_next_values = v_net(next_states)
                    tensor_advantages = rewards + tmdp.gamma*tensor_next_values*(1-done) - tensor_values
            

            dec_alpha = max(1e-8, alpha*(1 - episode/episodes))

            # Bound evaluation
            if( tmdp.tau > 0 and t % update_rate == 0):
                Q = np.minimum(q1_func.get_values(), q2_func.get_values())

                rel_pol_adv = compute_relative_policy_advantage_function(pi_ref, pi_old, Q)
                rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                delta = compute_delta(d, pi_old)
                pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                model_adv = compute_expected_model_advantage(rel_model_adv, delta)

                delta_U = get_sup_difference_U(U)
                #delta_U = get_sup_difference_Q(Q) # Valid only if reward associated to teleport is null
                if delta_U == 0:
                    delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                

                d_inf_pol = get_d_inf_policy(pi_ref, pi_old)
                d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
                d_exp_pol = get_d_exp_policy(pi_ref, pi_old, d)
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

                #print(optimal_pairs)
                #print(teleport_bounds)

                policy_pi.soft_update(ref_policy, alpha_star)
                ref_policy.soft_update(policy_pi, 1)
                tmdp.update_tau(tau_star)
                print("Alpha*: {} tau*: {} Episode: {} length: {} #teleports:{}".format(alpha_star, tau_star, episode, len(rewards),teleport_count))
                if r_sum > 0:
                    print("Got not null reward {}!".format(r_sum))
                if tau_star == 0:
                    print("Converged to the original problem, episode {}".format(episode))
                    convergence_t = episode
                    policy_pi.soft_update(ref_policy, 1)
        
            else:
                policy_pi.soft_update(ref_policy, 1.)
                print("Episode: {} length: {}".format(episode, len(rewards)))
                if r_sum > 0:
                    print("Got not null reward {}!".format(r_sum))
            
            reward_records.append(r_sum)

            # Reset the batch
            rewards = []   
            teleport_count = 0

    return {"convergence_t": convergence_t, "reward_records": reward_records}



def curriculum_PPO(tmdp:TMDP, policy_pi:ActorNet, ref_policy:ActorNet, value_func:ValueNet, 
                   q_func:QNet, loss_opt, q_opt, alpha=1., eps=.2, vf_coeff=0.5, 
                   h_coeff=0.1, episodes=5000, batch_size=1, biased=True):
    
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
    teleport_count = 0

    # Mid-term results
    reward_records = []
    states = []
    actions = []
    logit_list = []
    logprob_list = []
    next_states = []
    rewards = []
    flags_list = []
    t = 0
    for episode in range(episodes): # Each episode is a single time step
        
        # Sample the environment
        while True:
            s = tmdp.env.s
            a = policy_pi.act(s) # Select action from policy
            _, l, p = ref_policy.act_and_log_prob(s)
            s_prime, r, flags, _ =  tmdp.step(a)
            #print("State: ", s, "Action: ", a, "Next State: ", s_prime, "Reward: ", r, "Done: ", flags["done"])
            flags["terminated"] = terminated
            
            if not flags["teleport"]:
                states.append(s)
                actions.append(a)
                rewards.append(r)
                logit_list.append(l)
                logprob_list.append(p)
                next_states.append(s_prime)
                flags_list.append(flags)
                v_next = 0
                if flags["done"]:# or flags["teleport"]:
                    tmdp.reset()
                    batch_size += 1
                else:
                    with torch.no_grad():
                        v_next = value_func.get_value(s_prime)
            
                dec_alpha = max(1e-8, alpha*(1 - episode/episodes))
                U[s,a,s_prime] = r + tmdp.gamma*v_next
            else:
                teleport_count += 1
                batch_size += 1 # the reset is done in the teleportation
            
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
        if( (batch_size != 0 and batch_size % batch_size == 0) or done or terminated):
            
            # If the batch is empty, means that the first action occurred was a teleport, therefore nothing to learn
            if len(rewards) > 0: 
                r_sum = sum(rewards)
                cum_reward = np.zeros_like(rewards)
                reward_len = len(rewards)
                for i in reversed(range(reward_len)):
                    cum_reward[i] = rewards[i] + (tmdp.gamma*cum_reward[i+1] if i < reward_len-1 else rewards[i])

                # Combined Loss Optimization of Value function and Reference Policy

                tensor_states = torch.tensor(states, dtype=torch.long).to(value_func.device)
                tensor_actions = torch.tensor(actions, dtype=torch.long).to(q_func.device)
                tensor_next_states = torch.tensor(next_states, dtype=torch.long).to(q_func.device)
                
                tensor_logs_prob = torch.tensor(logprob_list, dtype=torch.float).to(ref_policy.device).unsqueeze(dim=1)
                tensor_cum_reward = torch.tensor(cum_reward, dtype=torch.float).to(value_func.device).unsqueeze(dim=1)
                
                tensor_values = value_func(tensor_states)
                tensor_logits_new = ref_policy(tensor_states)

                tensor_advantages = tensor_cum_reward - tensor_values # Compute advantages of visited states

                tensor_logs_prob_new = -F.cross_entropy(tensor_logits_new, tensor_actions, reduction='none').unsqueeze(dim=1)
                ratio = torch.exp(tensor_logs_prob_new - tensor_logs_prob)

                l_clip = torch.min(ratio*tensor_advantages, torch.clamp(ratio, 1-eps, 1+eps)*tensor_advantages)

                # Compute probabilities and log probabilities of next state for entropy calculation
                tensor_logits_next_state = ref_policy(tensor_next_states)
                next_state_prob = F.softmax(tensor_logits_next_state, dim=-1)
                next_state_log_prob = torch.log(next_state_prob)
                
                #h = next_state_prob.unsqueeze(dim=1) @ next_state_log_prob.unsqueeze(dim=2)
                
                vf_loss = F.mse_loss(tensor_values, tensor_cum_reward, reduction='none')
                
                loss = -l_clip + vf_loss * vf_coeff #- h*h_coeff
                loss_opt.zero_grad()
                loss.sum().backward()
                loss_opt.step()
                
                # QNet Optimization
                torch.cuda.empty_cache()
                tensor_q_values = q_func(tensor_states)
                tensor_q_values_actions = tensor_q_values.gather(1, tensor_actions.unsqueeze(dim=1))
                
                tensor_q_values_next = q_func(tensor_next_states)
                tensor_rewards = torch.tensor(rewards, dtype=torch.float).to(q_func.device).unsqueeze(dim=1)
                tensor_done = torch.tensor([flags["done"] for flags in flags_list], dtype=torch.float).to(q_func.device).unsqueeze(dim=1)
                
                exp_q = tensor_rewards + tmdp.gamma*tensor_q_values_next.max(dim=1)[0].unsqueeze(dim=1)* (1-tensor_done)
                
                # Minimize the MSE loss of (r + gamma* argmax_a' Q(s',a') - Q(s,a))
                q_loss = F.mse_loss(exp_q, tensor_q_values_actions, reduction='none')
                q_opt.zero_grad()
                q_loss.sum().backward()
                q_opt.step()
                
                
                pi_old = policy_pi.get_probabilities()
                pi_ref = ref_policy.get_probabilities()

                # Bound evaluation
                if( tmdp.tau > 0):
                    Q = q_func.get_values()
                    rel_pol_adv = compute_relative_policy_advantage_function(pi_ref, pi_old, Q)
                    rel_model_adv = compute_relative_model_advantage_function(tmdp.env.P_mat, tmdp.xi, U)
                    d = compute_d_from_tau(tmdp.env.mu, tmdp.env.P_mat, tmdp.xi, pi_old, tmdp.gamma, tmdp.tau)
                    delta = compute_delta(d, pi_old)
                    pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)
                    model_adv = compute_expected_model_advantage(rel_model_adv, delta)

                    delta_U = get_sup_difference_U(U)
                    #delta_U = get_sup_difference_Q(Q) # Valid only if reward associated to teleport is null
                    if delta_U == 0:
                        delta_U = (tmdp.env.reward_range[1]-tmdp.env.reward_range[0])/(1-tmdp.gamma)
                    

                    d_inf_pol = get_d_inf_policy(pi_ref, pi_old)
                    d_inf_model = get_d_inf_model(tmdp.env.P_mat, tmdp.xi)
                    d_exp_pol = get_d_exp_policy(pi_ref, pi_old, d)
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
                    policy_pi.soft_update(ref_policy, alpha_star)
                    tmdp.update_tau(tau_star)
                    print("Alpha*: {} tau*: {} Episode: {} length: {} #teleports:{}".format(alpha_star, tau_star, episode, len(rewards),teleport_count))
                    if r_sum > 0:
                        print("Got not null reward {}!".format(r_sum))
                    if tau_star == 0:
                        print("Converged to the original problem, episode {}".format(episode))
                        convergence_t = episode
                        policy_pi = ref_policy
            
                else:
                    policy_pi.soft_update(ref_policy, 1.)
                    print("Episode: {} length: {}".format(episode, len(rewards)))
                    if r_sum > 0:
                        print("Got not null reward {}!".format(r_sum))
                reward_records.append(r_sum)

            # Reset the batch
            states = []
            actions = []
            rewards = []
            next_states = []
            logit_list = []
            logprob_list = []
            flags_list = []
            batch_size = 0
            teleport_count = 0

    return {"policy_pi": policy_pi, "value_func":value_func, "q_func":q_func, "convergence_t": convergence_t, "reward_records": reward_records}
