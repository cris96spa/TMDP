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
from bound import *
from policy_utils import *
import matplotlib.pyplot as plt
import mlflow 
import os

class CurriculumMPI():

    def __init__(self, tmdp:TMDP, Q=None, theta=None, theta_ref=None, device=None, 
                 checkpoint=False, checkpoint_dir=None, checkpoint_name=None,
                 checkpoint_step:int=500):
        
        ######################################### Learning Quantities ###########################################
        self.tmdp = tmdp                                                                                        #                             
                                                                                                                #                         
        if Q is None:                                                                                           #                          
            Q = np.zeros((tmdp.nS, tmdp.nA))                                                                    #           
        self.Q = Q                                                                                              #
                                                                                                                #
        self.V = np.zeros(tmdp.nS)                                                                              #           
        self.U = np.zeros((tmdp.nS, tmdp.nA, tmdp.nS))                                                          #
                                                                                                                #
        if theta is None:                                                                                       #                                      
            theta = np.zeros((tmdp.nS, tmdp.nA))                                                                #                              
        self.theta = theta                                                                                      #
                                                                                                                # 
        if theta_ref is None:                                                                                   #  
            theta_ref = np.zeros((tmdp.nS, tmdp.nA))                                                            #    
        self.theta_ref = theta_ref                                                                              #                                        
                                                                                                                #
        if device is None:                                                                                      #                                      
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                               #
            if not torch.cuda.is_available():                                                                   #
                if torch.backends.mps.is_available():                                                           #
                    device = torch.device("mps")                                                                #   
        self.device = device                                                                                    #
                                                                                                                #
        ######################################### Training Parameters ###########################################
        self.k = 0                                  # number of episodes in the current trajectory              #
        self.t = 0                                  # number of episodes in the current batch                   #
        self.done = False                           # flag to indicate end the training                         #
        self.terminated = False                     # flag to indicate the forced termination of the training   #
        self.rewards = []                           # rewards for current trajectory                            #
        self.temp_decay = 0                         # temperature decay factor                                  #               
        self.lr_decay = 1                           # learning rate decay factor                                #
        self.episode = 0                            # episode counter                                           #
                                                                                                                #
        ######################################### Teleport Bound Parameters #####################################
        self.alpha_star = 1                         # PI learning rate                                          #        
        self.tau_star = 1                           # MI new value of tau                                       #
        self.teleport_count = 0                     # number of teleports during the batch                      #
                                                                                                                #
        ##########################################Lists and Trajectories ########################################
        self.batch = []                             # batch of trajectories                                     #
        self.traj = []                              # current trajectory                                        #
        self.reward_records = []                    # avg_rewards over each processed batch                     #      
        self.Qs = []                                # Q values during training                                  #
        self.temps = []                             # learning rates during training                            #
        self.thetas = []                            # policy parameters during training                         #    
        self.taus = []                              # taus values during training                               #
        ######################################### Checkpoint Parameters #########################################
        if checkpoint_dir is None:                                                                              #                                         
            checkpoint_dir = "./checkpoints"                                                                    #
        if checkpoint_name is None:                                                                             #                                            
            checkpoint_name =  tmdp.env.__class__.__name__+ "{}_{}".format(tmdp.nS, tmdp.nA)                    #
                                                                                                                #
        self.checkpoint = checkpoint                # flag to save checkpoints                                  #    
        self.checkpoint_dir = checkpoint_dir        # directory to save checkpoints                             #    
        self.checkpoint_name = checkpoint_name      # name of the checkpoint file                               #    
        self.checkpoint_step = checkpoint_step      # number of episodes to save a checkpoint                   #
        #########################################################################################################

    def train(self, model_lr:float=.25,pol_lr:float=.12,
              batch_size:int=1, temp:float=1., lam:float=0.,
              final_temp:float=0.02, episodes:int=5000,
              check_convergence:bool=False, epochs:int=1,
              biased:bool=False, 
              param_decay:bool=True, log_mlflow:bool=False,
              debug:bool=False, original_mu=None,
              max_length:int=0):
        """
            Curriculum MPI training and sample loop
        """
        self.tmdp.reset()                           # reset the environment

        
        ################################################## Parameter Initialization ##################################################
        self.biased = biased                        # flag to use biased or unbiased performance improvement bount
        self.episodes = episodes                    # number of episodes to train
        self.debug = debug                          # flag to print debug information
        ####################################### Additional Counters #######################################
        stucked_count = 0                           # number of batches updates without improvement
        self.max_length = self.episodes if max_length == 0 else max_length                  # maximum length of a trajectory
        
        # Tensor conversion
        if original_mu is None:
            original_mu = self.tmdp.env.mu
        self.tensor_mu = torch.tensor(original_mu, dtype=torch.float32).to(self.device)
        self.tensor_P_mat = torch.tensor(self.tmdp.env.P_mat, dtype=torch.float32).to(self.device)
        self.tensor_xi = torch.tensor(self.tmdp.xi, dtype=torch.float32).to(self.device)

        # Pre-Compute the D_inf distance metric
        self.d_inf_model = get_d_inf_model(self.tmdp.env.P_mat, self.tmdp.xi)
        r_sum = 0
        ################################################## Training and Sampling Loop ##################################################
        while self.episode < self.episodes:                                                 # loop over episodes
            
            s = self.tmdp.env.s                                                                 # current state from the environment
            policy = softmax_policy(self.theta[s], temperature=temp+self.temp_decay)       # get softmax policy
            
            ############################################## Sampling ############################################################
            flags = self.sample_step(policy)                                                # sample a step from the environment
            
            self.episode += 1                                                               # increment the episode counter

            if self.episode==self.episodes-1:                                               # if last episode   
                self.done = flags["done"]                                               # check if the episode is done
                self.terminated = not self.done
                print("Sampling loop is over. Done flag: {}, Terminated flag: {}".format(self.done, self.terminated))
                # If terminated last trajectory is inconsistent, therefore is discarded (if done, instead, already added in the sample_step function)
                
            ############################################# Checkpointing #############################################                     
            if self.episode % self.checkpoint_step == 0 or self.done or self.terminated:
                self.Qs.append(np.copy(self.Q))
                self.thetas.append(np.copy(self.theta))
                self.taus.append(self.tmdp.tau)
                if not debug and self.episode % min(100000, 10*self.checkpoint_step) == 0:
                    print("Episode: {} reward: {} tau {} batch_len {} teleports {}".format(self.episode, sum(self.rewards),self.tmdp.tau, len(self.batch), self.teleport_count))
                        
                if log_mlflow:
                    pass

                if self.checkpoint:
                    #self.save_checkpoint(episode)
                    pass
                if self.done or self.terminated:
                    break

            # Batch processing
            if( (len(self.batch) != 0 and len(self.batch) % batch_size == 0) or self.done or self.terminated):
                
                ############################################## Training ############################################################
                alpha_model = model_lr*self.lr_decay                                            # model learning rate    
                alpha_pol = pol_lr*self.lr_decay                                                # policy learning rate                   
                dec_temp = temp+self.temp_decay                                                 # temperature decay                                
                self.update(alpha_model, alpha_pol, dec_temp, lam, epochs)                      # Update Value Functions and Reference Policy                                                                        # train the model updating value functions and reference policy
                r_sum = sum(self.rewards)                                                       # sum of rewards in the batch


                ############################################# Bound evaluation #############################################
                s_time = time.time()                                                                            # start time    
                ref_policy = get_softmax_policy(self.theta_ref, temperature=dec_temp)                           # get softmax policy from reference policy
                self.V = compute_V_from_Q(self.Q, ref_policy)
                policy = get_softmax_policy(self.theta, temperature=dec_temp)                                   # get softmax policy from current policy
                optimal_pairs, teleport_bounds = self.bound_eval(ref_policy, policy)                            # get candidate pairs and the associated teleport bound value

                # Get the optimal values
                self.alpha_star, self.tau_star = get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds) # get the optimal values

                ########################################## Model and Policy Update ##########################################
                if debug:
                    print(optimal_pairs)
                    if self.alpha_star != 0 or self.tau_star != 0:                                                  # not null optimal values
                        print("Alpha*: {} tau*: {} Episode: {} reward: {} length: {} #teleports:{}".format(self.alpha_star, self.tau_star, self.episode, r_sum, len(self.rewards),self.teleport_count))
                    else:
                        print("No updates performed, episode: {} reward: {} length: {} #teleports:{}".format(self.episode, r_sum, len(self.rewards),self.teleport_count))

                if self.tau_star >= 0 and self.tau_star < self.tmdp.tau:    
                    if self.tau_star == 0:                                                   
                        print("Converged to the original problem, episode {}".format(self.episode))
                        self.convergence_t = self.episode                                                       # store the convergence episode                                                                                                                                 # if tau is not zero and not converged
                    self.tmdp.update_tau(self.tau_star)                                                         # Regular update without convergence
                    self.stucked_count = 0

                if self.alpha_star != 0:
                    self.theta = self.alpha_star*self.theta_ref + (1-self.alpha_star)*self.theta
                    self.stucked_count = 0
                
                e_time = time.time()                                                                            # end time  
                if debug:
                    print("Time for bound evaluation: ", e_time - s_time)
                
                ############################################# Decay Factors #############################################
                self.lr_decay = max(1e-8, 1-(self.episode)/(self.episodes)) if param_decay else 1                # learning rate decay
                self.temp_decay = (final_temp - temp)*(self.episode/self.episodes) if param_decay else 0         # temperature decay
                   
                ############################################# Preparing next batch #############################################
                self.batch = []                                         # reset the batch
                self.reward_records.append(r_sum)                       # append the sum of rewards to the records
                self.rewards = []                                       # reset the rewards list
                self.teleport_count = 0                                 # reset the teleport counter
                self.t = 0                                              # reset the episode counter in the batch    
                r_sum = 0                                               # reset the sum of rewards
                ############################################# Convergence Check #############################################
                if check_convergence and self.episode >= 0.25*self.episodes:
                    if self.alpha_star <= 0.1 and self.tmdp.tau <= 0.05:
                        stucked_count += 1
                        self.tmdp.tau*=0.99
                        if stucked_count > 50:
                            self.terminated = True
                            break

            if self.episode >= self.episodes: # Check Termination
                break

    def sample_step(self, policy):
        """
            Sample a step from the environment
        """
        s = self.tmdp.env.s                                         # current state from the environment
        a = select_action(policy)                                   # select action from the policy
        s_prime, r, flags, p = self.tmdp.step(a)                    # take a step in the environment            
        flags["terminated"] = self.terminated
        
        self.k += 1                                                 # increment the episode in the trajectory counter
        self.t += 1                                                 # increment the episode in batch counter
        sample = (s, a, r, s_prime, flags, self.t, self.k)          # sample tuple
        self.traj.append(sample)                                    # append sample to the trajectory           
        if len(self.traj) >= self.max_length:                       
            flags["terminated"] = True
            print("Trajectory length exceeded the maximum length, episode {}".format(self.episode))
        self.rewards.append(r)                                      # append reward to the rewards list   
            
        if flags["done"] or flags['terminated']:                                           # if terminal state is reached                              
            self.tmdp.reset()                                       # reset the environment
            self.batch.append(self.traj)                            # append the trajectory to the batch
            # reset current trajectory information
            self.traj = []
            self.k = 0
        if flags["teleport"]:                                       # if teleport happened
            self.teleport_count += 1                                # increment the teleport counter

        return flags

    def update(self, alpha_model, alpha_pol, dec_temp, lam, epochs=1):
        """
            Update the model using the collected batch of trajectories
        """
        for _ in range(epochs):                                         # loop over epochs
            if epochs > 1:                                              
                self.tmdp.env.np_random.shuffle(self.batch)             # shuffle the batch
            for traj in self.batch:                                     # loop over trajectories
                if lam!= 0:                                     
                    e = np.zeros((self.tmdp.nS, self.tmdp.nA))              # Reset eligibility traces at the beginning of each trajectory
                for j, sample in enumerate(traj):                       # loop over samples in the trajectory
                    
                    s, a, r, s_prime, flags, t, k = sample              # unpack sample tuple    
                
                    ##################################### Train Value Functions #####################################
                    if not flags["teleport"]:
                        if flags["done"] or flags["terminated"]:                                   # Terminal state reached
                            td_error = alpha_model*(r - self.Q[s, a])       # Consider only the reward
                        else:                                               # Regular state transition
                            #a_prime = traj[j+1][1]                          # get the next action     
                            a_prime = greedy(s_prime, self.Q, self.tmdp.env.allowed_actions[int(s_prime)])
                            td_error = alpha_model*(r + self.tmdp.gamma*self.Q[s_prime, a_prime]- self.Q[s, a]) 
                                                                        
                        if lam == 0 or not flags["done"] or not flags["terminated"]:
                            self.Q[s,a] += td_error                         # update Q values of the visited state-action pair
                        else:
                            e[s,a] = 1                                      # frequency heuristic with saturation  
                            self.Q += e*td_error                            # update all Q values with eligibility traces
                            e = self.tmdp.gamma*lam*e                       # recency heuristic 

                        ######################################### Compute the Advantage #########################################
                        ref_policy = softmax_policy(self.theta_ref[s],     # get softmax probabilities associated to the current state
                                                    temperature=dec_temp) 
                        
                        Vf_s = np.matmul(ref_policy, self.Q[s])             # compute the value function
                        A= self.Q[s,a] - Vf_s                              # compute advantage function
                        
                        ##################################### Compute U values #####################################
                        self.U[s,a,s_prime] += alpha_model*(r + self.tmdp.gamma*self.V[s_prime] - self.U[s,a,s_prime]) 

                        ######################################### Train Policy #########################################
                        # Computing Policy Gradient
                        g_log_pol = - ref_policy
                        g_log_pol[a] += 1
                        g_log_pol = g_log_pol/(dec_temp)
                        self.theta_ref[s] += alpha_pol*g_log_pol*A          # reference policy parameters update
                    else:
                        if lam != 0:
                            e = np.zeros((self.tmdp.nS, self.tmdp.nA))      # Reset eligibility traces if teleport happened
    def bound_eval(self, ref_policy, policy):
        """
            Evaluate the teleport bound for performance improvement
        """

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
        delta_U = get_sup_difference(tensor_U) 
        if delta_U == 0:
            delta_U = (1-self.tmdp.gamma**10)/(1-self.tmdp.gamma)
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
                                                        self.d_exp_model, self.tmdp.tau,self.tmdp.gamma,
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

    def state_dict(self):
        """
            Return the state dictionary
        """
        return {
            "Q": self.Q,
            "V": self.V,
            "U": self.U,
            "theta": self.theta,
            "theta_ref": self.theta_ref,
            "reward_records": self.reward_records,
            "Qs": self.Qs,
            "temps": self.temps,
            "thetas": self.thetas,
            "episode": self.episode,
            "lr_decay": self.lr_decay,
            "temp_decay": self.temp_decay,
            "taus": self.taus

        }

    def save_checkpoint(self):
        """
            Save the checkpoint
        """
        checkpoint = self.state_dict()
        torch.save(checkpoint, "{}/{}/{}.pth".format(self.checkpoint_dir, self.checkpoint_name, self.episode))
        print("Saved checkpoint at episode {}".format(self.episode))

    def load_checkpoint(self, episode):
        """
            Load the checkpoint
        """
        checkpoint = torch.load("{}/{}/{}.pth".format(self.checkpoint_dir, self.checkpoint_name, episode))
        self.Q = checkpoint["Q"]
        self.V = checkpoint["V"]
        self.U = checkpoint["U"]
        self.theta = checkpoint["theta"]
        self.theta_ref = checkpoint["theta_ref"]
        self.reward_records = checkpoint["reward_records"]
        self.Qs = checkpoint["Qs"]
        self.temps = checkpoint["temps"]
        self.thetas = checkpoint["thetas"]
        self.episode = checkpoint["episode"]
        self.lr_decay = checkpoint["lr_decay"]
        self.temp_decay = checkpoint["temp_decay"]
        self.taus = checkpoint["taus"]
        print("Loaded checkpoint at episode {}".format(episode))

    
    def save_model(self, path):
        """
            Save the model
        """
        torch.save(self.state_dict(), path)
        print("Saved model at {}".format(path))

    def save_to_mlflow(self):
        """
        Logs the model as an MLflow artifact.
        """
        # Define a temporary path to save the model
        temp_path = "./temp_model.pth"
        
        # Save the model using the existing save_model function
        self.save_model(temp_path)
        
        # Log the model file as an MLflow artifact
        mlflow.log_artifact(temp_path, "model")

        # Clean up: remove the temporary file after logging
        os.remove(temp_path)
        print("Model logged to MLflow and local file removed.")

    def load_model(self, path):
        """
            Load the model
        """
        checkpoint = torch.load(path)
        self.Q = checkpoint["Q"]
        self.V = checkpoint["V"]
        self.U = checkpoint["U"]
        self.theta = checkpoint["theta"]
        self.theta_ref = checkpoint["theta_ref"]
        self.reward_records = checkpoint["reward_records"]
        self.Qs = checkpoint["Qs"]
        self.temps = checkpoint["temps"]
        self.thetas = checkpoint["thetas"]
        self.episode = checkpoint["episode"]
        self.lr_decay = checkpoint["lr_decay"]
        self.temp_decay = checkpoint["temp_decay"]
        self.taus = checkpoint["taus"]
        print("Loaded model from {}".format(path))


    def load_model_from_mlflow(self, run_id, model_artifact_path):
        """
            Loads the model from an MLflow artifact given a run ID and artifact path.
        """

        # Construct the full path to the model artifact
        model_path = mlflow.get_artifact_uri(artifact_path=model_artifact_path, run_id=run_id)
        
        # Load the model using the custom loading function
        self.load_model(model_path)


