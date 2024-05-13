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
from TMDP import TMDP
from model_functions import *
from algorithms import *

class CurriculumMPI():

    def __init__(self, tmdp:TMDP, Q=None, theta=None, theta_ref=None, U=None, model_lr=.25, 
                 pol_lr=.12, episodes=5000, status_step=50000, batch_size=1, lam=0., epochs=1,
                 temp=1, final_temp=1e-3, device=None, biased=False, use_delta_Q=False):
        
        ######################################### Learning Quantities #########################################
        self.tmdp = tmdp

        if Q is None:
            Q = np.zeros((tmdp.nS, tmdp.nA))
        self.Q = Q

        if U is None:
            U = np.zeros((tmdp.nS, tmdp.nA))
        self.U = U

        if theta is None:
            theta = np.zeros((tmdp.nS, tmdp.nA))
        self.theta = theta

        if theta_ref is None:
            theta_ref = np.zeros((tmdp.nS, tmdp.nA))
        self.theta_ref = theta_ref

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model_lr = model_lr
        self.pol_lr = pol_lr
        self.temp = temp                            # temperature for softmax policy
        self.final_temp = final_temp                # final temperature for softmax policy
        self.temp_decay = 1                         # temperature decay factor       
        self.lr_decay = 1                           # learning rate decay factor                

        ######################################### Training Parameters #########################################
        self.episodes = episodes                    # number of episodes to run
        self.status_step = status_step              # Status information update step
        self.batch_size = batch_size                # Number of trajectories to collect before update
        self.epochs = epochs                        # Number of epochs to train each batch
        self.lam = lam                              # lambda for eligibility traces
        self.done = False                           # flag to indicate end the training
        self.terminated = False                     # flag to indicate the forced termination of the training

        ####################################### Teleport Bound Parameters #######################################
        self.alpha_star = 1                         # PI learning rate
        self.tau_star =1                            # MI new value of tau
        self.use_delta_Q = use_delta_Q              # flag to use delta Q instead of delta U for performance improvement bound
        self.biased = biased                        # flag to use biased or unbiased performance improvement bount

        ####################################### Additional Counters #######################################
        self.stucked_count = 0                      # number of batches updates without improvement
        self.k = 0                                  # number of episodes in the current trajectory
        self.t = 0                                  # number of episodes in the current batch
        self.teleport_count = 0                     # number of teleports during the batch

        ####################################### Lists and Trajectories #######################################
        self.batch = []                             # batch of trajectories
        self.traj = []                              # current trajectory          
        self.rewards = []                           # rewards for current trajectory
        self.reward_records = []                    # avg_rewards over each processed batch
        self.exp_performances = []                  # expected performances over each processed batch
        self.Qs = []                                # Q values during training
        self.Vs = []                                # V values during training
        self.lrs = []                               # learning rates during training
        self.thetas = []                            # policy parameters during training

    def sample_step(self, policy):
        """
            Sample a step from the environment
        """
        s = self.tmdp.env.s                         # current state from the environment
        a = select_action(policy[s])                # select action from the policy
        s_prime, r, flags, p = self.tmdp.step(a)    # take a step in the environment
        a_prime = select_action(policy[s_prime])    # select action of next state from the policy             
        flags["terminated"] = self.terminated
        sample = (s, a, r, s_prime, a_prime, flags, self.t, self.k) # sample tuple

        if not flags["teleport"]:                   # Following regular probability transitions function
            
            self.traj.append(sample)                # append sample to the trajectory           
            self.rewards.append(r)                  # append reward to the rewards list   
            self.k += 1                             # increment the episode in the trajectory counter
            self.t += 1                             # increment the episode in batch counter
            
            if flags["done"]:                       # if termina state is reached                              
                self.tmdp.reset()                   # reset the environment
                self.batch.append(self.traj)        # append the trajectory to the batch

                # reset current trajectory information
                self.traj = []
                self.k = 0
        else:                                       # Following teleportation distribution
            self.teleport_count += 1                # increment the teleport counter
            
            if len(self.traj) > 0:                  # current trajectory not empty
                self.batch.append(self.traj)        # append the trajectory to the batch
                
                # reset current trajectory information
                self.traj = []
                self.k = 0
        return sample

    def train(self):
        """
            Train the model using the collected batch of trajectories
        """
        for _ in range(self.epochs):                                            # loop over epochs
            for traj in self.batch:                                             # loop over trajectories
                
                e = np.zeros((self.tmdp.nS, self.tmdp.nA))                      # Reset eligibility traces at the beginning of each trajectory
                
                for sample in traj:                                             # loop over samples in the trajectory
                    
                    s, a, r, s_prime, a_prime, flags, t, k = sample             # unpack sample tuple    
                
                    ##################################### Train Value Functions #####################################
                    td_error = self.model_lr*self.lr_decay*(r +                 # temporal difference error
                                                            self.tmdp.gamma*self.Q[s_prime, a_prime] - self.Q[s, a])
                    
                    e[s,a] = 1                                                  # frequency heuristic with saturation                      
                    if self.lam == 0:
                        self.Q[s,a] += td_error*e[s,a]                          # update Q values of the visited state-action pair
                    else:
                        self.Q += e*td_error                                    # update all Q values with eligibility traces
                    
                    e *= self.tmdp.gamma*self.lam                               # recency heuristic 

                    ref_policy = get_softmax_policy(self.theta_ref,             # get softmax policy from reference policy
                                                temperature=self.temp*self.temp_decay) 
                
                    V = compute_V_from_Q(self.Q, ref_policy)                    # compute value function from Q values
                    A = self.Q[s,a] - V[s]                                      # compute advantage function
                    
                    self.U[s,a,s_prime] += self.model_lr*self.lr_decay*(r +     # update the model
                                                                    self.tmdp.gamma*V[s_prime] - self.U[s,a,s_prime]) 

                    
                    ######################################### Train Policy #########################################
                    # Computing Policy Gradient
                    g_log_pol = - ref_policy[s]
                    g_log_pol[a] += 1
                    g_log_pol = g_log_pol/(self.temp*self.temp_decay)
                    
                    self.theta_ref[s] += self.pol_lr*self.lr_decay*g_log_pol*A  # policy parameters update

                    #################################### Compute Expected Performance ####################################
                    tensor_V = torch.tensor(V, dtype=torch.float32).to(self.device)
                    self.exp_performances.append(compute_expected_j(tensor_V, self.tensor_mu))   # expected performance of the policy

    def bound_eval(self):
        """
            Evaluate the teleport bound for performance improvement
        """
        ref_policy = get_softmax_policy(self.theta_ref, temperature=self.temp*self.temp_decay)  # get softmax policy from reference policy
        policy = get_softmax_policy(self.theta, temperature=self.temp*self.temp_decay)          # get softmax policy from current policy

        # Tensor conversion
        tensor_ref_pol = torch.tensor(ref_policy, dtype=torch.float32).to(self.device)
        tensor_pol = torch.tensor(policy, dtype=torch.float32).to(self.device)
        tensor_Q = torch.tensor(self.Q, dtype=torch.float32).to(self.device)
        tensor_U = torch.tensor(self.U, dtype=torch.float32).to(self.device)

        # Compute Policy Advantages
        rel_pol_adv = compute_relative_policy_advantage_function(tensor_ref_pol, tensor_pol, tensor_Q)
        d = compute_d_from_tau(self.tensor_mu, self.tensor_P_mat, self.tensor_xi, tensor_pol, self.tmdp.gamma, self.tmdp.tau)
        self.pol_adv = compute_expected_policy_advantage(rel_pol_adv, d)                             # compute expected policy advantage
        
        
        # Compute Policy Distance Metric
        self.d_inf_pol = get_d_inf_policy(tensor_pol, tensor_ref_pol)
        self.d_exp_pol = get_d_exp_policy(tensor_pol, tensor_ref_pol, d)

        # Compute Delta U
        if self.use_delta_Q:
            delta_U = get_sup_difference(tensor_Q)
        else:
            delta_U = get_sup_difference(tensor_U) 
        if delta_U == 0:
            delta_U = (self.tmdp.env.reward_range[1]-self.tmdp.env.reward_range[0])/(1-self.tmdp.gamma)
        self.delta_U = delta_U

        # Compute Model Advantages
        if self.tmdp.tau > 0:
            delta = compute_delta(d, tensor_pol)
            rel_model_adv = compute_relative_model_advantage_function(self.tensor_P_mat, self.tensor_xi, tensor_U)
            model_adv = compute_expected_model_advantage(rel_model_adv, delta)                   # compute expected model advantage

            # Compute Model Distance Metric
            d_exp_model = get_d_exp_model(self.tensor_P_mat, self.tensor_xi, delta)
        else:
            model_adv = 0
            d_exp_model = 0
        
        self.model_adv = model_adv
        self.d_exp_model = d_exp_model

        # Compute teleport bound candidate pairs
        pairs = get_teleport_bound_optimal_values(self.pol_adv, self.model_adv, self.delta_U,
                                                        self.d_inf_pol, self.d_exp_pol, self.d_inf_model,
                                                        self.d_exp_model, self.tmdp.tau, self.tmdp.gamma,
                                                        biased=self.biased)
        teleport_bounds = []
        
        # Compute teleport bound for candidate pairs
        for alpha_prime, tau_prime in pairs:
            bound = compute_teleport_bound(alpha_prime, self.tmdp.tau, tau_prime, self.pol_adv, 
                                           self.model_adv, self.tmdp.gamma, 
                                           self.d_inf_pol, self.d_inf_model,
                                           self.d_exp_pol, self.d_exp_model, 
                                           self.delta_U, biased=self.biased)
            teleport_bounds.append(bound)
        
        # Get the optimal values
        self.alpha_star, self.tau_star = get_teleport_bound_optima_pair(pairs, teleport_bounds)
        
        return pairs, teleport_bounds

    def main_loop(self):
        """
            Curriculum MPI training and sample loop
        """
        ################################################## Parameter Initialization ##################################################
        episode = 0

        # Tensorize the environment for PyTorch
        self.tensor_mu = torch.tensor(self.tmdp.env.mu, dtype=torch.float32).to(self.device)         
        self.tensor_P_mat = torch.tensor(self.tmdp.env.P_mat, dtype=torch.float32).to(self.device)
        self.tensor_xi = torch.tensor(self.tmdp.xi, dtype=torch.float32).to(self.device)
        
        # Compute the D_inf distance metric
        self.d_inf_model = get_d_inf_model(self.tmdp.env.P_mat, self.tmdp.xi)

        ################################################## Training and Sampling Loop ##################################################
        while episode < self.episodes:                                                              # loop over episodes
            
            policy = get_softmax_policy(self.theta, temperature=self.temp*self.temp_decay)        # get softmax policy
            
            ############################################## Sampling ############################################################
            sample = self.sample_step(policy)                                                       # sample a step from the environment
            episode += 1                                                                            # increment the episode counter

            if episode==self.episodes-1:                                                            # if last episode   
                
                if self.alpha_star > 0 and self.tmdp.tau < 1e-4 and self.stucked_count == 0:
                    self.episodes += max(100000, self.episodes * 0.10)                              # increase the number of episodes
                    print("Increasing the number of episodes to {} ".format(self.episodes))
                else:
                    self.done = sample[5]["done"]                                                   # check if the episode is done
                    self.terminated = not self.done
                    print("Sampling loop is over. Done flag: {}, Terminated flag: {}".format(self.done, self.terminated))

            
            # Batch processing
            if( (len(self.batch) > 0 and len(self.batch) % self.batch_size == 0) or self.done or self.terminated):
                ############################################## Training ############################################################
                self.train()                                                                        # train the model updating value functions and reference policy
                r_sum = sum(self.rewards)                                                           # sum of rewards in the batch
                
                if r_sum > 0:
                    print("Got not null reward {}!".format(r_sum))

                s_time = time.time()                                                                # start time
                ####################################### Performance Improvement Bound ##################################################
                pairs, teleport_bound = self.bound_eval()                                                                                                        # evaluate the bound for performance improvement
                print("Candidate pairs: {}".format(pairs))

                if self.alpha_star != 0 or self.tau_star != 0:
                    print("Alpha*: {} tau*: {} Episode: {} length: {} #teleports:{}".format(self.alpha_star, self.tau_star, episode, len(self.rewards), self.teleport_count))
                else:
                    print("No updates performed, episode: {} length: {} #teleports:{}".format(episode, len(self.rewards),self.teleport_count))
                

                if self.tau_star == 0 and self.tmdp.tau != 0:                                       # if converged to the original problem
                    print("Converged to the original problem, episode {}".format(episode))          # to be done only once
                    self.convergence_t = episode
                    self.tmdp.update_tau(self.tau_star)                                             # update the transition function of the model
                    self.stucked_count = 0                                                          # reset the stucked counter
                elif self.tmdp.tau > 0:
                    self.tmdp.update_tau(self.tau_star)
                    self.stucked_count = 0

                if self.alpha_star != 0:
                    theta = self.alpha_star*self.theta_ref + (1-self.alpha_star)*theta              # update the policy parameters following the teleport bound and the reference policy
                    self.stucked_count = 0

                e_time = time.time()                                                                # end time
                print("Time for bound evaluation: ", e_time - s_time)   
            
                self.lr_decay = max(1e-8, 1-(episode)/(self.episodes))
                self.temp_decay = self.temp + (self.final_temp - self.temp)*(episode/self.episodes)

                self.batch = []                                                                     # reset the batch
                self.reward_records.append(r_sum)                                                   # append the sum of rewards to the records