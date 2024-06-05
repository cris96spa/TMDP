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
from policy_utils import *
import matplotlib.pyplot as plt
import mlflow 
import os

class CurriculumPPO():

    def __init__(self, tmdp:TMDP, theta=None, theta_ref=None, device=None, 
                 checkpoint=False, checkpoint_dir=None, checkpoint_name=None,
                 checkpoint_step:int=500):
        
        ######################################### Learning Quantities ###########################################
        self.tmdp = tmdp                                                                                        #                        
        self.V = np.zeros(tmdp.nS)                                                                              #           
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
        self.teleport_count = 0                     # number of teleports during the batch                      #
                                                                                                                #
        ##########################################Lists and Trajectories ########################################
        self.batch = []                             # batch of trajectories                                     #
        self.traj = []                              # current trajectory                                        #
        self.reward_records = []                    # avg_rewards over each processed batch                     #      
        self.Vs = []                                # V values during training                                  #
        self.temps = []                             # learning rates during training                            #
        self.thetas = []                            # policy parameters during training                         #    
        self.taus = []                              # taus values during training                               #
                                                                                                                #
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
                 epochs:int=10, eps_ppo:float=0.2, eps_model:float=0.2,
                 param_decay:bool=True,max_length:int=0, 
                 entropy_coef:float=0.1, debug:bool=False):
        """
            Curriculum MPI training and sample loop
        """
        self.tmdp.reset()                           # reset the environment

        
        ################################################## Parameter Initialization ##################################################
        self.episodes = episodes                                                            # number of episodes to train
        self.max_length = self.episodes if max_length == 0 else max_length                  # maximum length of a trajectory
        self.final_temp = final_temp                                                        # final temperature
        self.entropy_coef = entropy_coef                                                    # entropy coefficient
        self.debug = debug                                                                  # debug flag
        if self.tmdp.tau != 0:                                                              # if the model is already the original model
            self.n_updates = compute_n(self.tmdp.gamma, self.tmdp.tau, eps_model)           # number of updates to reach the original model
            self.update_rate = int(self.episodes/self.n_updates)                            # update rate in terms of number of episode between two updates
        self.update_counter = 0                                                             # update counter
        ####################################### Additional Counters #######################################
        
        # Tensorize the environment for PyTorch
        # Tensor conversion
        self.tensor_mu = torch.tensor(self.tmdp.env.mu, dtype=torch.float32).to(self.device)

        ################################################## Training and Sampling Loop ##################################################
        while self.episode < self.episodes:                                                 # loop over episodes
            
            s = self.tmdp.env.s                                                             # current state from the environment
            policy = softmax_policy(self.theta[s], temperature=temp+self.temp_decay)        # get softmax policy
            
            ############################################## Sampling ############################################################
            flags = self.sample_step(policy)                                             # sample a step from the environment
            
            self.episode += 1                                                               # increment the episode counter
            if self.tmdp.tau != 0:
                if self.episode % self.update_rate == 0:                                    # update the model
                    self.update_counter += 1
            
            if self.episode==self.episodes-1:                                               # if last episode   
                self.done = flags["done"]                                                   # check if the episode is done
                self.terminated = not self.done
                print("Sampling loop is over. Done flag: {}, Terminated flag: {}".format(self.done, self.terminated))
                # If terminated last trajectory is inconsistent, therefore is discarded (if done, instead, already added in the sample_step function)
           
            # Batch processing
            if( (len(self.batch) != 0 and len(self.batch) % batch_size == 0) or self.done or self.terminated):
                s_time = time.time()                                                        # start time
                ############################################## Training ############################################################
                alpha_model = model_lr*self.lr_decay                                          # model learning rate    
                alpha_pol = pol_lr*self.lr_decay                                              # policy learning rate                   
                dec_temp = temp+self.temp_decay                                               # temperature decay                                
                
                self.update(alpha_model, alpha_pol, dec_temp, lam, epochs=epochs, eps_ppo=eps_ppo)  # Update Value Functions and Reference Policy
                
                r_sum = sum(self.rewards)                                                  # sum of rewards in the batch
                e_time = time.time()                                                      # end time
                
                if debug:   
                    print("Batch Processing time time: {}".format(e_time-s_time))
                
                ############################################# Model Update #############################################  
                self.update_model(eps_model=eps_model)                                                                # update the model
                if debug:
                    print("Episode: {} reward: {} length: {} #teleports:{} update_counter: {}".format(self.episode, r_sum, len(self.rewards),self.teleport_count, self.update_counter))
                e_time = time.time()                                                          
            
                
                ############################################# Decay Factors #############################################
                self.lr_decay = max(1e-8, 1-(self.episode)/(self.episodes)) if param_decay else 1                # learning rate decay
                self.temp_decay = (final_temp - temp)*(self.episode/self.episodes) if param_decay else 0         # temperature decay
                   
                ############################################# Preparing next batch #############################################
                self.batch = []                                         # reset the batch
                self.reward_records.append(r_sum)                       # append the sum of rewards to the records
                self.rewards = []                                       # reset the rewards list
                self.teleport_count = 0                                 # reset the teleport counter
                self.t = 0                                              # reset the episode counter in the batch    
                self.update_counter = 0                                 # reset the update counter
                            
            ############################################# Checkpointing #############################################                     
            
            if (self.episode % self.checkpoint_step == 0) or self.done or self.terminated:
                self.Vs.append(np.copy(self.V))
                self.thetas.append(np.copy(self.theta))
                self.taus.append(self.tmdp.tau)
                
                if not debug and self.episode % (10*self.checkpoint_step) == 0:
                    print("Episode: {} reward: {} length: {}".format(self.episode, r_sum, len(self.rewards)))

                if self.checkpoint:
                    #self.save_checkpoint(episode)
                    pass
                if self.done or self.terminated:
                    break

            if self.episode >= self.episodes: # Check Termination
                break

    def sample_step(self, policy):
        """
            Sample a step from the environment
        """
        s = self.tmdp.env.s                                             # current state from the environment
        a = select_action(policy)                                    # select action from the policy
        s_prime, r, flags, p = self.tmdp.step(a)                        # take a step in the environment            
        flags["terminated"] = self.terminated
        
        self.k += 1                                                 # increment the episode in the trajectory counter
        self.t += 1                                                 # increment the episode in batch counter
        sample = (s, a, r, s_prime, flags, self.t, self.k)          # sample tuple
        self.traj.append(sample)                                    # append sample to the trajectory           
        if len(self.traj) >= self.max_length:                       
            flags["terminated"] = True
        self.rewards.append(r)                                      # append reward to the rewards list   
            
            
        if flags["done"] or flags["terminated"]:                    # if terminal state is reached                              
            self.tmdp.reset()                                       # reset the environment
            self.batch.append(self.traj)                            # append the trajectory to the batch
            # reset current trajectory information
            self.traj = []
            self.k = 0
        if flags["teleport"]:                                       # if teleport happened
            self.teleport_count += 1                                # increment the teleport counter

        return flags


    def update(self, alpha_model, alpha_pol, dec_temp, lam, epochs=1, eps_ppo=0.2):
        """
            Update the model using the collected batch of trajectories
        """

        # get softmax policy from current policy, used for exploration
        old_policy = get_softmax_policy(self.theta, temperature=dec_temp)
        self.compute_gae(lam, self.tmdp.gamma)
        
        for _ in range(epochs):                                         # loop over epochs
            if epochs > 1:                                              
                self.tmdp.env.np_random.shuffle(self.batch)             # shuffle the batch
            for traj in self.batch:                                     # loop over trajectories
                for j, sample in enumerate(traj):                       # loop over samples in the trajectory
                    
                    s, a, r, s_prime, flags, t, k, A = sample              # unpack sample tuple    

                    if not flags["teleport"]:                               # Following regular probability transitions function
                        ##################################### Train Value Functions #####################################
                        if flags["done"]:                               # Terminal state reached or teleportation happened
                            td_error = alpha_model*(r - self.V[s])      # Consider only the reward
                        else:                                               # Regular state transition
                            td_error = alpha_model*(r + self.tmdp.gamma*self.V[s_prime]- self.V[s]) 
                                            
                        #if lam == 0 or not flags["done"]:
                        self.V[s] += td_error                         # update Q values of the visited state-action pair
                        
                        """else:
                            e[s,a] = 1                                      # frequency heuristic with saturation
                            self.V += e*td_error                            # update all Q values with eligibility traces
                            e *= self.tmdp.gamma*lam"""                        # recency heuristic 


                        ######################################### Compute the Advantage #########################################
                        ref_policy = softmax_policy(self.theta_ref[s],     # get softmax probabilities associated to the current state
                                                    temperature=dec_temp) 
                        
                        
                        ######################################### Train Policy #########################################
                        # Using logarithm for numerical stability
                        ref_log_pol = np.log(ref_policy[a]+ 1e-8)           # compute the log policy from the reference policy
                        old_log_pol = np.log(old_policy[s,a] + 1e-8)        # compute the log policy from the current policy
                        ratio = np.exp(ref_log_pol - old_log_pol)           # compute the ratio between the two policies                    
                        
                        l_clip = np.clip(ratio, 1-eps_ppo, 1+eps_ppo)       # compute the clipped ratio
                        surr_1 = ratio*A                                    # compute the surrogate function 1
                        surr_2 = l_clip*A                                   # compute the surrogate function 2

                        # Computing Policy Gradient
                        if ratio > 1-eps_ppo and ratio < 1+eps_ppo:         # UPDATE
                            g_log_pol = - ref_policy                        # compute the gradient of the log policy
                            g_log_pol[a] += 1
                            g_log_pol = g_log_pol/dec_temp
                            """if A > 0:
                                print(f"Positive Advantage {A}, picked a good action {a} in state {s}, V[s]: {self.V[s]} V[s']: {self.V[s_prime]} reward: {r}")
                                print(f"Terminated {flags['terminated']}, done {flags['done']}")
                                print(f"theta_ref: {self.theta_ref[s]}")
                                print(f"g_log_pol: {g_log_pol}, surr_1: {surr_1}, surr_2: {surr_2}")
                                policy_entropy = self.entropy(ref_policy)           # compute the entropy of the policy    
                                entropy_bonus = self.entropy_coef*policy_entropy    # compute the entropy bonus
                                objective = min(surr_1, surr_2) + entropy_bonus     # compute the objective function
                                print(f"Objective: {objective}")
                                print(f"entropy_bonus: {entropy_bonus}")"""
                            
                        elif A > 0:                                         # NO UPDATE
                            g_log_pol = 0                                   # if the advantage is positive, the gradient is zero
                            
                        else:                                               # Negative Advantage Update
                            g_log_pol = - ref_policy                        # compute the gradient of the log policy
                            g_log_pol[a] += 1
                            g_log_pol = g_log_pol/dec_temp
                            
                        policy_entropy = self.entropy(ref_policy)                       # compute the entropy of the policy    
                        entropy_bonus = self.entropy_coef*policy_entropy*self.lr_decay  # compute the entropy bonus
                        objective = min(surr_1, surr_2) + entropy_bonus                 # compute the objective function
                        self.theta_ref[s] += alpha_pol*g_log_pol*objective              # reference policy parameters update
                    else:
                        pass                                                   # Teleport happened 
                        """if lam!= 0:                                     
                            e = np.zeros((self.tmdp.nS, self.tmdp.nA))"""
        #ref_pol = get_softmax_policy(self.theta_ref, temperature=self.final_temp)
        #self.V = compute_V_from_Q(self.Q, ref_pol)                      # update the value function
        self.theta = self.theta_ref                            # update the policy parameters with the reference policy parameters    

    def update_model(self, eps_model:float=0.2):
        """
            Update the model probability transition function
        """
        if self.tmdp.tau > 0 and self.update_counter > 0:
            
            eps_n = eps_model*self.update_counter
            tau_prime = compute_tau_prime(self.tmdp.gamma, self.tmdp.tau, eps_n)
            if self.debug:
                print("Updating the model from tau: {} to tau_prime: {}".format(round(self.tmdp.tau, 6), (round(tau_prime, 6))))
            if tau_prime == 0:
                print("Convergence to the original model in {} steps".format(self.episode))
            self.tmdp.update_tau(tau_prime)

    def compute_gae(self, lam, gamma):
        
        for traj in self.batch:
            last_adv = 0
            adv = 0
            for i in reversed(range(len(traj))):
                s, a, r, s_prime, flags, t, k = traj[i]
                
                if not flags["teleport"]:
                    if flags["done"]:                                   # Terminal state reached
                        delta = r - self.V[s]                                
                        adv = last_adv = delta                          # Consider only the reward  
                    else:
                        delta = r + gamma*self.V[s_prime] - self.V[s]   # Compute the temporal difference
                        adv = last_adv = delta + gamma*lam*last_adv     # GAE advantage
                else:
                    adv = last_adv = 0                                  # Reset the advantage to zero
                traj[i] = (s, a, r, s_prime, flags, t, k, adv)          # Update the trajectory with the advantage

    def entropy(self, policy):
        """
            Compute the entropy of the policy
        """
        return -np.sum(policy * np.log(policy + 1e-8))

    def state_dict(self):
        """
            Return the state dictionary
        """
        return {
            "V": self.V,
            "theta": self.theta,
            "theta_ref": self.theta_ref,
            "reward_records": self.reward_records,
            "Vs": self.Vs,
            "temps": self.temps,
            "thetas": self.thetas,
            "taus": self.taus,
            "episode": self.episode,
            "lr_decay": self.lr_decay,
            "temp_decay": self.temp_decay,

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
        self.V = checkpoint["V"]
        self.theta = checkpoint["theta"]
        self.theta_ref = checkpoint["theta_ref"]
        self.reward_records = checkpoint["reward_records"]
        self.Vs = checkpoint["Vs"]
        self.temps = checkpoint["temps"]
        self.thetas = checkpoint["thetas"]
        self.taus = checkpoint["taus"]
        self.episode = checkpoint["episode"]
        self.lr_decay = checkpoint["lr_decay"]
        self.temp_decay = checkpoint["temp_decay"]
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
        self.V = checkpoint["V"]
        self.theta = checkpoint["theta"]
        self.theta_ref = checkpoint["theta_ref"]
        self.reward_records = checkpoint["reward_records"]
        self.Vs = checkpoint["Vs"]
        self.temps = checkpoint["temps"]
        self.thetas = checkpoint["thetas"]
        self.taus = checkpoint["taus"]
        self.episode = checkpoint["episode"]
        self.lr_decay = checkpoint["lr_decay"]
        self.temp_decay = checkpoint["temp_decay"]
        print("Loaded model from {}".format(path))


    def load_model_from_mlflow(self, run_id, model_artifact_path):
        """
            Loads the model from an MLflow artifact given a run ID and artifact path.
        """
        # Construct the full path to the model artifact
        model_path = mlflow.get_artifact_uri(artifact_path=model_artifact_path, run_id=run_id)
        
        # Load the model using the custom loading function
        self.load_model(model_path)
