import numpy as np
from gymnasium import Env
from DiscreteEnv import DiscreteEnv
from TMDP import TMDP
from model_functions import *
from gymnasium.utils import seeding
import torch
import torch.nn as nn
from torch.nn import functional as F
import constants
import time
from ReplayBuffer import ReplayBuffer

# Reproducibility
seed = None

def get_current_seed():
    global seed
    if seed is None:
        set_policy_seed(constants.SEEDS[0])
    return seed

def set_policy_seed(policy_seed):
    global np_random
    global seed
    np_random, seed = seeding.np_random(policy_seed)
    print("Current seed for result reproducibility: {}".format(seed))

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
    Test policies on the original problem
    Args:
        - tmdp (TMDP): the teleporting MDP
        - thetas (ndarray): collection of parameter vectors associated to the policies
        - episodes (int): number of episodes to run
        - temp (float): temperature value
    return (ndarray): the average reward collected over trajectories for each policy
"""
def test_policies(tmdp:TMDP, thetas, episodes=100, temp=1e-4):    
    rewards = []
    tau = tmdp.tau
    
    for theta in thetas:
        pi = get_policy(get_softmax_policy(theta, temperature=temp))
        tmdp.reset()
        tmdp.update_tau(0.)
        episode = 0
        cum_r = 0
        traj_count = 0
        while episode < episodes:
            s = tmdp.env.s
            a = select_action(pi[s])
            s_prime, reward, flags, prob = tmdp.step(a)
            cum_r += reward
            if flags["done"]:
                tmdp.reset()
                traj_count += 1
            episode += 1
        cum_r = cum_r/traj_count if traj_count > 0 else cum_r
        rewards.append(cum_r)
    
    tmdp.update_tau(tau)
    return rewards