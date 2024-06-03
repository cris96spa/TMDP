import numpy as np
from gymnasium import Env
from scipy.special import softmax
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model_functions import *


"""
    Compute the expected policy advantage \mathcal{A}_pi'_pi = sum_s d(s) * A_pi'_pi(s)
    Args:
        - rel_policy_adv (np.ndarray): the relative policy advantage function [nS]
        - d (np.ndarray): the discounted state distribution as a vector [nS]
    return (float): the expected policy advantage as a scalar value
"""
def compute_expected_policy_advantage(rel_policy_adv, d):

    if torch.is_tensor(rel_policy_adv): # Tensor version
        pol_adv = (d * rel_policy_adv).sum().item()
    else: # Numpy version
        pol_adv = np.matmul(d, np.transpose(rel_policy_adv))
    return pol_adv

"""
    Compute the expectation under discounted distribution of the relative model advantage \mathcal{A}_P_xi = sum_s sum_a delta(s,a) * A_P_xi(s,a)
    Args:
        - rel_model_adv (np.ndarray): the relative model advantage function [nS, nA]
        - delta (np.ndarray): the discount state action distribution under policy pi [nS, nA]
    return (float): the discounted distribution relative model advantage function as a scalar
"""
def compute_expected_model_advantage(rel_model_adv, delta):

    if torch.is_tensor(rel_model_adv): # Tensor version
        model_adv = (delta * rel_model_adv).sum().item()
    else: # Numpy version
        model_adv = 0
        nS, _ = delta.shape
        for s in range(nS):
            model_adv +=  np.matmul(delta[s], np.transpose(rel_model_adv[s]))

    return model_adv