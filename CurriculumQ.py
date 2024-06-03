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

class CurriculumQ():

    def __init__(self, tmdp:TMDP, Q=None, device=None, 
                 checkpoint=False, checkpoint_dir=None, checkpoint_name=None,
                 checkpoint_step:int=50000):
        
        ######################################### Learning Quantities ###########################################
        self.tmdp = tmdp                                                                                        #                             
                                                                                                                #                         
        if Q is None:                                                                                           #                          
            Q = np.zeros((tmdp.nS, tmdp.nA))                                                                    #           
        self.Q = Q                                                                                              #                       
                                                                                                                #
        if device is None:                                                                                      #                                      
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                               #   
        self.device = device                                                                                    #
                                                                                                                #
        ######################################### Training Parameters ###########################################
        self.k = 0                                  # number of episodes in the current trajectory              #
        self.t = 0                                  # number of episodes in the current batch                   #
        self.done = False                           # flag to indicate end the training                         #
        self.terminated = False                     # flag to indicate the forced termination of the training   #
        self.rewards = []                           # rewards for current trajectory                            #
        self.lr_decay = 1                           # learning rate decay factor                                #
        self.exp_rate_decay = 1                     # exploration rate decay factor                             #
        self.episode = 0                            # episode counter                                           #
                                                                                                             #
        ######################################### Teleport Bound Parameters #####################################
        self.teleport_count = 0                     # number of teleports during the batch                      #
                                                                                                                #
        ##########################################Lists and Trajectories ########################################
        self.batch = []                             # batch of trajectories                                     #
        self.traj = []                              # current trajectory                                        #
        self.reward_records = []                    # avg_rewards over each processed batch                     #      
        self.Qs = []                                # Q values during training                                  #                                 
        self.taus = []                              # tau values during training                                #
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

    def train(self, model_lr:float=.25,
              batch_size:int=1, lam:float=0., episodes:int=5000,
              exp_rate:float=0.4, eps_model:float=0.2,
              param_decay:bool=True, log_mlflow:bool=False,
              debug:bool=False):
        """
            Curriculum MPI training and sample loop
        """
        self.tmdp.reset()                                                                   # reset the environment

        
        ################################################## Parameter Initialization ##################################################
        self.episodes = episodes                                                            # number of episodes to train
        if self.tmdp.tau != 0:                                                              # if the model is already the original model
            self.n_updates = compute_n(self.tmdp.gamma, self.tmdp.tau, eps_model)           # number of updates to reach the original model
            self.update_rate = int(self.episodes/self.n_updates)                            # update rate in terms of number of episode between two updates
        self.debug = debug                                                                  # debug flag
        self.update_counter = 0
        ####################################### Additional Counters #######################################
        
        # Tensor conversion
        self.tensor_mu = torch.tensor(self.tmdp.env.mu, dtype=torch.float32).to(self.device)

        ################################################## Training and Sampling Loop ##################################################
        while self.episode < self.episodes:                                                 # loop over episodes
            
            ############################################## Sampling ############################################################
            eps = exp_rate*self.exp_rate_decay                                              # exploration rate
            
            flags = self.sample_step(eps)                                                   # sample a step from the environment
            
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
                alpha_model = model_lr*self.lr_decay                                        # model learning rate    
                
                self.update(alpha_model, lam)                                               # Update Value Functions and Reference Policy                                                                        # train the model updating value functions and reference policy
                
                r_sum = sum(self.rewards)                                                   # sum of rewards in the batch
                
                
                e_time = time.time()                                                            # end time
                if debug:
                    print("Batch Processing time time: {}".format(e_time-s_time))
                ############################################# Model Update #############################################  
                self.update_model(eps_model=eps_model)                                                                # update the model
                if debug:
                    print("Episode: {} reward: {} length: {} #teleports:{}".format(self.episode, r_sum, len(self.rewards),self.teleport_count))
                e_time = time.time()                                                          
            
                
                ############################################# Decay Factors #############################################
                self.lr_decay = max(1e-8, 1-(self.episode)/(self.episodes)) if param_decay else 1           # learning rate decay
                self.exp_rate_decay = max(0, 1- (self.episode/self.episodes)**2) if param_decay else 1      # temperature decay
                   
                ############################################# Preparing next batch #############################################
                self.batch = []                                         # reset the batch
                self.reward_records.append(r_sum)                       # append the sum of rewards to the records
                self.rewards = []                                       # reset the rewards list
                self.teleport_count = 0                                 # reset the teleport counter
                self.t = 0                                              # reset the episode counter in the batch    
                self.update_counter = 0                                 # reset the update counter
            ############################################# Checkpointing #############################################                     
            if (self.episode % self.checkpoint_step == 0) or self.done or self.terminated:
                self.Qs.append(np.copy(self.Q))
                self.taus.append(self.tmdp.tau)
                
                if not debug and self.episode % (10*self.checkpoint_step) == 0:
                    print("Episode: {} reward: {} length: {}".format(self.episode, r_sum, len(self.rewards)))
                if log_mlflow:
                    pass

                if self.checkpoint:
                    #self.save_checkpoint(episode)
                    pass
                if self.done or self.terminated:
                    break

    def sample_step(self, eps):
        """
            Sample a step from the environment
        """
        s = self.tmdp.env.s                                             # current state from the environment
        allowed_actions = self.tmdp.env.allowed_actions[int(s)]         # allowed actions in the current state
        a = eps_greedy(s, self.Q, eps, allowed_actions)                 # select action from the policy
        s_prime, r, flags, p = self.tmdp.step(a)                        # take a step in the environment            
        flags["terminated"] = self.terminated
        
        self.k += 1                                                 # increment the episode in the trajectory counter
        self.t += 1                                                 # increment the episode in batch counter
        sample = (s, a, r, s_prime, flags, self.t, self.k)# sample tuple
        self.traj.append(sample)                                    # append sample to the trajectory           
        self.rewards.append(r)                                      # append reward to the rewards list   
            
            
        if flags["done"]:                                           # if terminal state is reached                              
            self.tmdp.reset()                                       # reset the environment
            self.batch.append(self.traj)                            # append the trajectory to the batch
            # reset current trajectory information
            self.traj = []
            self.k = 0
        if flags["teleport"]:                                       # if teleport happened
                self.teleport_count += 1                            # increment the teleport counter

        return flags


    def update(self, alpha_model, lam):
        """
            Update the model using the collected batch of trajectories
        """
        
        for traj in self.batch:                                     # loop over trajectories
            if lam!= 0:                                     
                e = np.zeros((self.tmdp.nS, self.tmdp.nA))          # Reset eligibility traces at the beginning of each trajectory
            
            for j, sample in enumerate(traj):                       # loop over samples in the trajectory
                
                s, a, r, s_prime, flags, t, k = sample              # unpack sample tuple    
            
                ##################################### Train Value Functions #####################################
                if not flags["teleport"]:                           # Regular transition function
                    if flags["done"]:
                        td_error = alpha_model*(r - self.Q[s,a])    # compute the TD error
                    else:
                        a_prime = greedy(s_prime, self.Q, self.tmdp.env.allowed_actions[int(s_prime)]) 
                        td_error = alpha_model*(r + self.tmdp.gamma*self.Q[s_prime, a_prime] - self.Q[s,a]) 
                                                                    
                    if lam == 0 or not flags["done"]:
                        self.Q[s,a] += td_error                     # update Q values of the visited state-action pair
                    else:
                        e[s,a] = 1                                  # frequency heuristic with saturation
                        self.Q += e*td_error                        # update all Q values with eligibility traces
                        e *= self.tmdp.gamma*lam                    # recency heuristic 
                else:
                    if lam!= 0:                                     
                            e = np.zeros((self.tmdp.nS, self.tmdp.nA))  # Reset eligibility traces if teleport happens
    
    def update_model(self, eps_model:float=0.2, adaptive:bool=True, tuning_rate:float=0.95):
        """
            Update the model probability transition function
        """
        if self.tmdp.tau > 0 and self.update_counter > 0:
            
            """if adaptive: # Compute the eps_model threshold that lead convergence to the original model in remaining steps
                if self.episode > self.episodes*(tuning_rate - (1-tuning_rate)) or self.tmdp.tau < 0.15: # Dynamic tuning part
                    remaining_steps = max(1, self.episodes*tuning_rate - self.episode)
                    eps_model = compute_eps_model(self.tmdp.gamma, self.tmdp.tau, remaining_steps)""" 
            
            eps_n = eps_model*self.update_counter
            
            tau_prime = compute_tau_prime(self.tmdp.gamma, self.tmdp.tau, eps_n)
            if self.debug:
                print("Updating the model from tau: {} to tau_prime: {}".format(round(self.tmdp.tau, 6), (round(tau_prime, 6))))
            if tau_prime == 0:
                print("Convergence to the original model in {} steps".format(self.episode))
            self.tmdp.update_tau(tau_prime)
            

    def state_dict(self):
        """
            Return the state dictionary
        """
        return {
            "Q": self.Q,
            "reward_records": self.reward_records,
            "Qs": self.Qs,
            "episode": self.episode,
            "lr_decay": self.lr_decay,
            "exp_rate_decay": self.exp_rate_decay,
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
        self.reward_records = checkpoint["reward_records"]
        self.Qs = checkpoint["Qs"]
        self.episode = checkpoint["episode"]
        self.lr_decay = checkpoint["lr_decay"]
        self.exp_rate_decay = checkpoint["exp_rate_decay"]
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
        self.reward_records = checkpoint["reward_records"]
        self.Qs = checkpoint["Qs"]
        self.episode = checkpoint["episode"]
        self.lr_decay = checkpoint["lr_decay"]
        self.exp_rate_decay = checkpoint["exp_rate_decay"]
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

