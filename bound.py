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


######################### Difference Metrics #########################
"""
    Compute the superior difference between any two elements of the Q function
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
    return (float): the superior difference between any two elements of the state action value function Q
"""
def get_sup_difference(value_function):
    if torch.is_tensor(value_function):
        return (torch.max(value_function) - torch.min(value_function)).item()
    return np.max(value_function) - np.min(value_function)



"""
    Compute the superior of the l1 norm of the difference between two policies
    Args:
        - pi (np.ndarray): the policy [nS, nA]
        - pi_prime (np.ndarray): the new policy [nS, nA]
    return (float): the superior of the l1 norm of the difference between two policies
"""
def get_d_inf_policy(pi, pi_prime):

    if torch.is_tensor(pi):
        l1_norm = torch.sum(torch.abs(pi - pi_prime), dim=1)
        d_inf = torch.max(l1_norm)
        return d_inf.item()
    
    l1_norm = np.sum(np.abs(pi - pi_prime), axis=1)
    d_inf = np.max(l1_norm)
    return d_inf

"""
    Compute the superior of the l1 norm of the difference between two probability transition functions
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - xi (np.ndarray): teleport probability distribution [nS]
    return (float): the superior of the l1 norm of the difference between two probability transition functions
"""
def get_d_inf_model(P_mat, xi):

    if torch.is_tensor(P_mat):
        nS, nA = P_mat.shape[0], P_mat.shape[1]
        Xi = xi.unsqueeze(1) * torch.ones((nS, nA, nS)).to(xi.device)
        l1_norm = torch.sum(torch.abs(P_mat - Xi), dim=2)
        max_over_actions = torch.max(l1_norm, dim=1).values
        d_inf = torch.max(max_over_actions)
        return d_inf.item()
    
    nS, nA, _ = P_mat.shape
    Xi = np.tile(xi, (nA, nS)).T
    Xi = Xi.reshape((nS, nA, nS))
    l1_norm = np.sum(np.abs(P_mat - Xi), axis=2)
    max_over_actions = np.max(l1_norm, axis=1)
    d_inf = np.max(max_over_actions)
    return d_inf

"""
    Compute the expected value of the l1 norm of the difference among two policies
    Args:
        - pi (np.ndarray): the policy [nS, nA]
        - pi_prime (np.ndarray): the new policy [nS, nA]
        - d (np.ndarray): the discounted state distribution as a vector [nS]
    return (float): the expected value of the l1 norm of the difference among two policies
"""
def get_d_exp_policy(pi, pi_prime, d):

    if torch.is_tensor(pi):
        l1_norm = torch.sum(torch.abs(pi - pi_prime), dim=1)
        d_exp = torch.matmul(d, l1_norm)
        return d_exp.item()
    
    l1_norm = np.sum(np.abs(pi - pi_prime), axis=1)
    d_exp = np.matmul(d, np.transpose(l1_norm))
    return d_exp

"""
    Compute the expected value of the l1 norm of the difference among two probability transition functions
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - xi (np.ndarray): teleport probability distribution [nS]
        - delta (np.ndarray): the discount state action distribution under policy pi [nS, nA]
    return (float): the expected value of the l1 norm of the difference among two probability transition functions
"""
def get_d_exp_model(P_mat, xi, delta):

    if torch.is_tensor(P_mat):
        nS, nA = P_mat.shape[0], P_mat.shape[1]
        Xi = xi.unsqueeze(0).unsqueeze(1).expand(nS,nA,nS) # shape [1, 1, nS]
        l1_norm = torch.sum(torch.abs(P_mat - Xi), dim=2)
        d_exp = torch.sum(delta * l1_norm)
        return d_exp.item()
    
    nS, nA, _ = P_mat.shape
    Xi = np.tile(xi, (nA, nS)).T
    Xi = Xi.reshape((nS, nA, nS))

    l1_norm = np.sum(np.abs(P_mat - Xi), axis=2)
    d_exp = 0
    for s in range(nS):
        d_exp += np.matmul(delta[s], np.transpose(l1_norm[s]))
    return d_exp


######################### Teleport Bound #########################
"""
    Compute the teleport bound B(alpha, tau_prime) = adv - bias - diss_penalty
    Args:
        - alpha (float): the step size
        - tau (float): the teleport probability
        - tau_prime (float): the new teleport probability
        - policy_adv (np.ndarray): the policy advantage function [nS, nA]
        - model_adv (np.ndarray): the model advantage function [nS, nA]
        - gamma (float): discount factor
        - d_inf_policy (float): the superior of the l1 norm of the difference between two policies
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
        - d_exp_policy (float): the expected value of the l1 norm of the difference among two policies
        - d_exp_model (float): the expected value of the l1 norm of the difference among two probability transition functions
        - delta_U (float): the difference between the two value functions
    return (float): the teleport lower bound for performance improvement
"""
def compute_teleport_bound(alpha, tau, tau_prime, policy_adv, 
                           model_adv, gamma, d_inf_policy, 
                           d_inf_model, d_exp_policy, d_exp_model, delta_U, biased=True
                           ):
    adv = (alpha * policy_adv + (tau- tau_prime) * model_adv)/(1-gamma)
    bias = 0
    if biased:
        bias = gamma*(tau+tau_prime)*d_exp_model/(1-gamma)
    diss_penalty = gamma*delta_U/(2*(1-gamma)**2) * (alpha**2*d_exp_policy*d_inf_policy 
                                                      + alpha*abs(tau-tau_prime)*d_exp_policy*d_inf_model 
                                                      + alpha*abs(tau-tau_prime)*d_exp_model*d_inf_policy 
                                                      + gamma*(tau-tau_prime)**2*d_exp_model*d_inf_model
                                                      )
    #print("alpha: {}, tau: {}, tau_prime: {}, adv: {}, bias: {}, diss_penalty: {}".format(alpha, tau, tau_prime, adv, bias, diss_penalty))
    return adv - diss_penalty - bias

"""
    Optimal value of alpha for tau_prime= tau
    Args:
        - policy_adv (np.ndarray): the policy advantage function [nS, nA]
        - gamma (float): discount factor
        - delta_U (float): the difference between the two value functions
        - d_exp_policy (float): the expected value of the l1 norm of the difference among two policies
        - d_inf_policy (float): the superior of the l1 norm of the difference between two policies
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
    return (float): the optimal value of alpha for tau_prime= tau
"""
def compute_alpha_tau(policy_adv, gamma, delta_U, d_exp_policy, d_inf_policy):
    alpha_tau = (1-gamma)* policy_adv/(gamma*delta_U*d_exp_policy*d_inf_policy)
    return round(alpha_tau, 5)


"""
    Optimal value of alpha for tau_prime= 0
    Args:
        - model_adv (np.ndarray): the model advantage function [nS, nA]
        - gamma (float): discount factor
        - delta_U (float): the difference between the two value functions
        - d_exp_model (float): the expected value of the l1 norm of the difference among two probability transition functions
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
    return (float): the optimal value of alpha for tau_prime= 0
"""
def compute_alpha_0(policy_adv, tau, gamma, delta_U, d_exp_policy, d_inf_policy, d_exp_model, d_inf_model):
    adv = (1-gamma)* policy_adv/(gamma*delta_U*d_exp_policy*d_inf_policy)
    diss_penalty = tau/2 * (d_inf_model/d_inf_policy + d_exp_model/d_exp_policy)
    alpha_0 =  adv - diss_penalty
    return round(alpha_0, 5)

"""
    Optimal value of tau_prime for alpha= 0
    Args:
        - model_adv (np.ndarray): the model advantage function [nS, nA]
        - gamma (float): discount factor
        - delta_U (float): the difference between the two value functions
        - d_exp_model (float): the expected value of the l1 norm of the difference among two probability transition functions
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
    return (float): the optimal value of tau_prime for alpha= 0
"""
def compute_tau_prime_0(tau, model_adv, gamma, d_exp_model, delta_U, d_inf_model, biased=True):
    
    common_terms = (1-gamma)/(gamma**2*delta_U*d_exp_model*d_inf_model) 
    bias = 0 if not biased else gamma*d_exp_model
    tau_prime = tau - (model_adv + bias)*common_terms

    return round(tau_prime, 5)

"""
    Optimal value of tau_prime for alpha= 1
    Args:
        - policy_adv (np.ndarray): the policy advantage function [nS, nA]
        - model_adv (np.ndarray): the model advantage function [nS, nA]
        - gamma (float): discount factor
        - delta_U (float): the difference between the two value functions
        - d_exp_policy (float): the expected value of the l1 norm of the difference among two policies
        - d_inf_policy (float): the superior of the l1 norm of the difference between two policies
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
        - d_exp_model (float): the expected value of the l1 norm of the difference among two probability transition functions
    return (float): the optimal value of tau_prime for alpha= 1
"""
def compute_tau_prime_1(tau, model_adv, gamma, d_exp_model, delta_U, d_inf_model, d_inf_policy, d_exp_policy, biased=True):
    
    common_terms = (1-gamma)/(gamma**2*delta_U*d_exp_model*d_inf_model) 
    bias = 0 if not biased else gamma*d_exp_model
    diss = 1/(2*gamma) * (d_exp_policy/d_exp_model + d_inf_policy/d_inf_model)
    tau_prime = tau - (model_adv + bias)*common_terms + diss

    return round(tau_prime, 5)

def get_teleport_bound_optimal_values(pol_adv, model_adv, delta_U, d_inf_pol, d_exp_pol,
                                       d_inf_model, d_exp_model, tau, gamma, biased=True):
    optimal_values = []
    
    if pol_adv > 0:
        # policy evaluation temrs
        if d_inf_pol != 0 and d_exp_pol != 0:
            # optimal value for alpha with tau'=tau
            alpha_tau = compute_alpha_tau(pol_adv, gamma, delta_U, d_exp_pol, d_inf_pol)
            if alpha_tau >= 0:
                optimal_values.append((min(1,round(alpha_tau, 5)), round(tau, 5)))
            
            # optimal value for alpha with tau'=0
            alpha_0 = compute_alpha_0(pol_adv, tau, gamma, delta_U, d_exp_pol, d_inf_pol, d_exp_model, d_inf_model)
            if alpha_0 >= 0 and alpha_0 <= 1:
                optimal_values.append((min(1, round(alpha_0,5)), 0.))

    if model_adv > 0:
        # model evaluation terms
        if d_inf_model != 0 and d_exp_model != 0:
            # optimal value for tau' with alpha=0
            tau_prime_0 = compute_tau_prime_0(tau, model_adv, gamma, d_exp_model, delta_U, d_inf_model, biased=biased)
            if tau_prime_0 >= 0 and tau_prime_0 <= 1:
                optimal_values.append((0., round(tau_prime_0, 5)))
            
            # optimal value for tau' with alpha=1
            tau_prime_1 = compute_tau_prime_1(tau, model_adv, gamma, d_exp_model, delta_U, d_inf_model, d_inf_pol, d_exp_pol, biased=biased)     
            if tau_prime_1 >= 0 and tau_prime_1 <= 1:
                optimal_values.append((1., round(tau_prime_1, 5)))
    else:
        pass

    if len(optimal_values) == 0:
        tau_prime = tau if tau > 0.1 else tau*0.5
        optimal_values.append((0., round(tau_prime, 5))) # No valid pairs found
    
    return optimal_values

def get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds, threshold=1e-6):
    alpha_star, tau_star = optimal_pairs[stochastic_argmax(teleport_bounds)]
    if tau_star < threshold:
        tau_star = 0
    return (alpha_star, tau_star)


"""
    Compute the teleport bound B(alpha, tau) associated to the pair (alpha=alpha_0, tau_prime=tau)
    Args:
        - tau (float): the teleport probability
        - policy_adv (np.ndarray): the policy advantage function [nS, nA]
        - gamma (float): discount factor
        - d_inf_policy (float): the superior of the l1 norm of the difference between two policies
        - d_exp_policy (float): the expected value of the l1 norm of the difference among two policies
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
        - d_exp_model (float): the expected value of the l1 norm of the difference among two probability transition functions
        - delta_U (float): the difference between the two value functions
"""
def compute_teleport_bound_alpha_tau(tau, policy_adv, 
                           gamma, d_inf_policy, 
                           d_exp_policy, d_exp_model, delta_U, biased=True
                           ):
    
    adv = policy_adv**2/(2*gamma*delta_U*d_exp_policy*d_inf_policy)
    bias = 0
    if biased:
        bias = 2*gamma*tau*d_exp_model/(1-gamma)
    return adv-bias

