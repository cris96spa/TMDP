import numpy as np
from gymnasium import Env
from scipy.special import softmax
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

######################### Model Functions #########################

"""
    Compute the average reward when picking action a in state s. It is evaluated as R(s,a) = sum_s' P(s'|s,a) * R(s,a,s')
    
    Args:    
        - P_mat (np.ndarray): probability transition function [nS, nA, nS] or state teleport probability distribution [nS]
        - reward (np.ndarray): the reward function [nS, nA, nS]
    return (np.ndarray):the average reward when picking action a in state s as a as [ns, nA] matrix
"""
def compute_r_s_a(P_mat, reward):
    # Average reward when taking action a in state s, of nS |S|x|A|
    nS, nA, _ = reward.shape
    # If the probability transition function is the teleport probability vector, we simply replicate it to match the shape of the reward function
    if len(P_mat.shape) == 1:
        P_mat = np.tile(P_mat, (nA, nS)).T
        P_mat = P_mat.reshape((nS, nA, nS))
        
    r_s_a =np.zeros(shape=(nS, nA))
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                r_s_a[s,a] += P_mat[s][a][s_prime] * reward[s, a, s_prime]
    return r_s_a

"""
    Extract the policy from a given state action value function
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy

    return (np.ndarray): the greedy policy according to Q, as [nS, nA]
"""
def get_policy(Q, det=True):
    pi = np.zeros(Q.shape)
    if det:
        for x in range(Q.shape[0]):
            pi[x,np.argmax(Q[x])] = 1
    else:
        for x in range(Q.shape[0]):
            pi[x] = softmax(Q[x]) 
            #pi[x] = Q[x]/np.sum(Q[x])
    return pi

"""
    Compute the probability of moving from state s to state sprime, under policy pi. It is evaluated as P(s'|s) = sum_a pi(a|s) * P(s'|s,a)
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - pi (np.ndarray): the given policy [nS, nA]

    return (np.ndarray): the probability of moving from state s to state sprime under policy pi [nS, nS] 
"""
def compute_transition_kernel(P_mat, xi, tau, pi):
   
    if torch.is_tensor(P_mat): # Tensor version
        
        P_mat_tau = (1-tau)*P_mat + tau*xi.unsqueeze(0).unsqueeze(1)
        nS, nA = P_mat.shape[0], P_mat.shape[1]
        P_sprime_s = torch.einsum('san,sa->sn', P_mat_tau, pi)
        return P_sprime_s
    else: # Numpy version
        xi_exp = np.expand_dims(np.expand_dims(xi, 0), 1)
        P_mat_tau = (1-tau)*P_mat + tau*xi_exp
        P_sprime_s = np.einsum('san,sa->sn', P_mat_tau, pi)
        return P_sprime_s

######################### Value Functions #########################
"""
    Extract the value function from a given state action value function
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - det (bool): deterministic flag. Whether or not considering a deterministic policy
    return (np.ndarray): the value function as V(s) = sum_a Q(s,a) * pi(a|s) [nS]
"""
def compute_V_from_Q(Q, pi):

    if torch.is_tensor(Q): # Tensor version
        return (Q * pi).sum(dim=1)
    else: # Numpy version
        V = np.sum(Q*pi, axis=1)
        return V


"""
    Extract the state action value function from a given state action nextstate value function Q = sum_s' P(s'|s,a) * U(s,a,s')
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - U (np.ndarray): state action next state value function as[nS, nA, nS]
    return (np.ndarray): the state action value function [nS, nA] 
"""
def compute_Q_from_U(P_mat, U):
    nS, nA, _ = U.shape
    Q = np.zeros((nS, nA))
    
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                Q[s, a] +=  U[s, a, s_prime] * P_mat[s][a][s_prime]
    return Q

"""
    Compute the state action nextstate value function U(s, a, s') = R(s,a,s') + gamma * V(s')
    Args:
        - reward (np.ndarray): reward function [nS, nA, nS]
        - gamma (float): discount factor
        - V (np.ndarray): the value function [nS]
    return (np.ndarray): the state action nextstate value function [nS, nA, nS]
"""
def compute_U_from_V(reward, gamma, V):

    if torch.is_tensor(reward): # Tensor version
        V_exp = V.unsqueeze(0).unsqueeze(0)
        U = reward + gamma*V_exp
    else: # Numpy version
        U = reward + gamma*V.reshape(1, 1, -1)
    return U


######################### Advantage Functions #########################
"""
    Compute the state policy advantage function A(s,a) = Q(s,a) - V(s)
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - V (np.ndarray): the value function [nS]
    return (np.ndarray): the policy advantage function [nS, nA]
"""
def compute_policy_advantage_function(Q, V):
    nS, nA = Q.shape
    pol_adv = np.zeros((nS, nA))
    for s in range(nS):
        pol_adv[s] = Q[s] - V[s]
    return pol_adv

"""
    Compute the model advantage function A(s, a, s') = U(s, a, s') - Q(s, a)
    Args:
        - U (np.ndarray): the state action nextstate value function [nS, nA, nS]
        - Q (np.ndarray): the state action value function [nS, nA]
    return (np.ndarray): the model advantage function [nS, nA, nS]
"""
def compute_model_advantage_function(U, Q):
    nS, nA = Q.shape
    model_adv = np.zeros((nS, nA, nS))

    for s in range(nS):
        for a in range(nA):
            model_adv[s, a] = U[s, a] - Q[s, a]
    return model_adv

"""
    Compute the relative policy advantage function A_pi_pi_prime(s)
    Args:
        - pi_prime (np.ndarray): the new policy to be compared [nS, nA]
        - A (np.ndarray): the policy advantage function [nS, nA]
    return (np.ndarray): the relative policy advantage function [nS]
"""
def compute_relative_policy_advantage_function(pi_prime, pi, Q,):

    if torch.is_tensor(pi_prime): # Tensor version
        delta_pol = pi_prime - pi
        rel_pol_adv = (delta_pol * Q).sum(dim=1)
    else: # Numpy version
        nS, _ = pi.shape
        rel_pol_adv = np.zeros(nS)
        for s in range(nS):
            delta_pol = pi_prime[s] - pi[s]
            rel_pol_adv[s] = np.matmul(delta_pol, np.transpose(Q[s]))
    
    return rel_pol_adv

"""
    Compute the relative policy advantage function A_pi_pi_prime(s) = sum_a pi_prime(a|s) * A(s,a)
    Args:
        - pi_prime (np.ndarray): the new policy to be compared [nS, nA]
        - A (np.ndarray): the policy advantage function [nS, nA]
    return (np.ndarray): the relative policy advantage function [nS]
"""
def compute_relative_policy_advantage_function_v2(pi_prime, A):
    nS, _ = pi_prime.shape
    rel_pol_adv = np.zeros(nS)
    for s in range(nS):
        rel_pol_adv[s] = np.matmul(pi_prime[s], np.transpose(A[s]))
    return rel_pol_adv


"""
    Compute the relative model advantage function A_P_xi(s,a) = sum_s' P(s'|s,a)-xi(s') * U(s,a,s')
    Args:
        - P_mat (nd.array): the probability transition function of the original problem [nS, nA, nS]
        - xi (np.ndarray): state teleport probability distribution [nS]
        - U (np.ndarray): state action next state value function [nS, nA, nS]
    return (np.ndarray): the relative model advantage function [nS, nA]
"""
def compute_relative_model_advantage_function(P_mat, xi, U):

    if torch.is_tensor(P_mat): # Tensor version
        Xi = xi.unsqueeze(0).unsqueeze(0) # shape [1, 1, nS]
        delta_model = P_mat - Xi
        rel_model_adv =(delta_model * U).sum(dim=2)
    else: # Numpy version
        nS, nA, _ = U.shape
        rel_model_adv = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                delta_model = P_mat[s][a] - xi
                rel_model_adv[s, a] = np.matmul(delta_model, np.transpose(U[s, a]))
    return rel_model_adv

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
######################### Performance Metrics #########################
"""
    Compute the theoretical discounted sum of returns as J = 1/(1-gamma) sum_s d(s) * sum_a pi(a|s) * R(s,a)
    Args:    
        - r_s_a (np.ndarray): the average reward when picking action a in state s [nS, nA]
        - pi (np.ndarray): the given policy [nS, nA]
        - d (np.ndarray): the discounted state distribution as a vector [nS]
        - gamma (float): discount factor
    return (float): the theoretical discounted sum of returns as a scalar value
"""
def compute_j(r_s_a, pi, d, gamma):
    nS, _ = pi.shape
    J = 0
    for s in range(nS):
        pol_reward = np.matmul(pi[s], np.transpose(r_s_a[s]))
        J += d[s]*pol_reward
    return J/(1-gamma)

"""
    Compute the expected return of the policy pi as J = sum_s mu(s) * V(s)
    Args:
        - V (np.ndarray): the value function [nS]
        - mu (np.ndarray): initial state distribution [nS]
    return (float): the expected return associated to the value function V and the initial state distribution mu
"""
def compute_expected_j(V, mu ):
    if torch.is_tensor(V):
        return torch.matmul(mu, V).item()
    return np.matmul(mu, np.transpose(V))

######################### Discounted State Distribution #########################

"""
    Compute the discounted state distribution as d = (1-gamma) * mu * (I - gamma*(1-tau)*P_mat - gamma*tau*Xi)^-1
    Args:
        - mu (np.ndarray): initial state distribution [nS]
        - P_mat (np.ndarray): probability transition function of the original problem [nS, nA, nS]
        - xi (np.ndarray): state teleport probability distribution [nS]
        - pi (np.ndarray): the given policy [nS, nA]
        - gamma (float): discount factor
        - tau (float): teleport probability
    return (np.ndarray): the discount state distribution as a vector [nS]
"""
def compute_d_from_tau(mu, P_mat, xi, pi, gamma, tau, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    if torch.is_tensor(P_mat): # Tensor version
        return compute_d_from_tau_tensor(mu, P_mat, xi, pi, gamma, tau, device)
    else: # Numpy version
        nS, nA = pi.shape
        I = np.eye(nS)
        P_sprime_s = compute_transition_kernel(P_mat, xi, tau, pi)

        inv = np.linalg.inv(I - gamma*P_sprime_s)
        d = np.matmul((1-gamma)*mu, inv)
        return d

def compute_d_from_tau_tensor(mu, P_mat, xi, pi, gamma, tau, device):
    nS, nA = pi.shape
    I = torch.eye(nS).to(device)
    P_sprime_s = compute_transition_kernel(P_mat, xi, tau, pi)
    # Ensure P_sprime_s is a valid stochastic matrix
    
    inv = torch.inverse(I - gamma*P_sprime_s)
    d = torch.matmul((1-gamma)*mu, inv)
    return d

"""
    Compute the gradient of the discounted state distribution as grad_d = (1-gamma) * mu * (I - gamma*((1-tau)*P_s_sprime + tau*Xi))^-1 * (Xi - P_sprime_s) * (I - gamma*((1-tau)*P_s_sprime + tau*Xi))^-1
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - xi (np.ndarray): state teleport probability distribution [nS]
        - mu (np.ndarray): initial state distribution [nS]
        - pi (np.ndarray): the given policy [nS, nA]
        - gamma (float): discount factor
        - tau (float): teleport probability
"""
def compute_grad_d_from_tau(P_mat, xi, mu, pi, gamma, tau):
    nS, _ = pi.shape
    I = np.eye(nS)
    P_sprime_s = compute_transition_kernel(P_mat, xi, tau, pi)

    Xi = np.tile(xi, nS).reshape((nS, nS))
    model_diff = Xi-P_sprime_s

    M = np.linalg.inv(I - gamma*((1-tau)*P_sprime_s + tau*Xi))
    
    grad_d = np.matmul(M, model_diff)
    grad_d = np.matmul(grad_d, M)
    grad_d = np.matmul(mu, grad_d)

    return (1-gamma)*gamma*grad_d


"""
    Compute the discounted state action distribution under policy pi as delta = pi * d
    Args:
        - d (np.ndarray) : the discounted state distribution as a vector [nS]
        - pi (np.ndarray): the given policy [nS, nA]    
    return (np.ndarray): the discount state action distribution under policy pi as [nS, nA]
"""
def compute_delta(d, pi):
    delta = pi * d[:, None]
    if torch.is_tensor(d):
        assert torch.all(delta >= 0), "State action distribution contains negative values (Tensor)"
    else:
        assert np.all(delta >= 0), "State action distribution contains negative values (Numpy)" 
    return delta

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
        print(f"Model advantage is negative {model_adv}")
        print("d_inf_model: {}, d_exp_model: {}".format(d_inf_model, d_exp_model))

    if len(optimal_values) == 0:
        print("No valid pairs found")
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



"""
    Compute the teleport bound B(0, 0) associated to the pair (alpha=alpha_0, tau_prime=0)
    Args:
        - tau (float): the teleport probability
        - policy_adv (np.ndarray): the policy advantage function [nS, nA]
        - model_adv (np.ndarray): the model advantage function [nS, nA]
        - gamma (float): discount factor
        - d_inf_policy (float): the superior of the l1 norm of the difference between two policies
        - d_exp_policy (float): the expected value of the l1 norm of the difference among two policies
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
        - d_exp_model (float): the expected value of the l1 norm of the difference among two probability transition functions
        - delta_U (float): the difference between the two value functions
    return (float): the teleport lower bound for performance improvement
"""
def compute_teleport_bound_alpha_0(tau, policy_adv, model_adv, gamma, d_inf_policy, d_exp_policy, 
                                   d_inf_model, d_exp_model, delta_U):
    
    adv = policy_adv**2/(2*gamma*delta_U*d_exp_policy*d_inf_policy) + tau*model_adv/(1-gamma)
    bias = gamma*tau*d_exp_model/(1-gamma)
    diss_penalty = tau*policy_adv/(2*(1-gamma))*(d_inf_model/d_inf_policy + d_exp_model/d_exp_policy)
    diss_penalty += gamma*tau**2*delta_U/(2*(1-gamma)**2)*(gamma*d_exp_model*d_inf_model - (d_inf_policy*d_exp_model + d_inf_model*d_exp_policy)**2/(4*d_inf_policy*d_exp_policy))
    return adv-bias - diss_penalty

def compute_teleport_bound_alpha_0_test(tau, policy_adv, model_adv, gamma, d_inf_policy, d_exp_policy, 
                                   d_inf_model, d_exp_model, delta_U):
    return (d_exp_model**2*d_inf_policy**2*delta_U**2*gamma**2*tau**2 - 4*d_exp_model*d_exp_policy*d_inf_model*d_inf_policy*delta_U**2*gamma**3*tau**2 + 2*d_exp_model*d_exp_policy*d_inf_model*d_inf_policy*delta_U**2*gamma**2*tau**2 + 8*d_exp_model*d_exp_policy*d_inf_policy*delta_U*gamma**3*tau - 8*d_exp_model*d_exp_policy*d_inf_policy*delta_U*gamma**2*tau + 4*d_exp_model*d_inf_policy*delta_U*gamma**2*policy_adv*tau - 4*d_exp_model*d_inf_policy*delta_U*gamma*policy_adv*tau + d_exp_policy**2*d_inf_model**2*delta_U**2*gamma**2*tau**2 + 4*d_exp_policy*d_inf_model*delta_U*gamma**2*policy_adv*tau - 4*d_exp_policy*d_inf_model*delta_U*gamma*policy_adv*tau - 8*d_exp_policy*d_inf_policy*delta_U*gamma**2*model_adv*tau + 8*d_exp_policy*d_inf_policy*delta_U*gamma*model_adv*tau + 4*gamma**2*policy_adv**2 - 8*gamma*policy_adv**2 + 4*policy_adv**2)/(8*d_exp_policy*d_inf_policy*delta_U*gamma*(gamma - 1)**2)


"""
    Compute the teleport bound B(0, tau) associated to the pair (alpha=0, tau_prime=tau_prime_0)
    Args:
        - tau (float): the teleport probability
        - model_adv (np.ndarray): the model advantage function [nS, nA]
        - gamma (float): discount factor
        - d_inf_model (float): the superior of the l1 norm of the difference between two probability transition functions
        - d_exp_model (float): the expected value of the l1 norm of the difference among two probability transition functions
        - delta_U (float): the difference between the two value functions
    return (float): the teleport lower bound for performance improvement
"""
def compute_teleport_bound_0_tau(tau, model_adv, gamma, d_inf_model, d_exp_model, delta_U, biased=True):
    adv = model_adv**2/(2*gamma**2 *delta_U*d_exp_model*d_inf_model) 
    bias = 0 if not biased else 2*gamma*tau*d_exp_model/(1-gamma) - model_adv/(gamma*delta_U*d_inf_model) - d_exp_model/(2*delta_U*d_inf_model)
    return adv-bias


"""
    Compute the teleport bound B(1, tau) associated to the pair (alpha=1, tau_prime=tau)
"""
def compute_teleport_bound_1_tau(tau, policy_adv, model_adv, gamma, d_inf_model, 
                                 d_inf_policy, d_exp_model, d_exp_policy, delta_U):
    
    adv = model_adv**2/(2*gamma**2 *delta_U*d_exp_model*d_inf_model) + d_exp_model/(2*delta_U*d_inf_model)
    adv += policy_adv/(1-gamma) + delta_U/(8*(1-gamma)**2*d_inf_model*d_exp_model)*(d_inf_model*d_exp_policy + d_inf_policy*d_exp_model)**2
    bias = 2*gamma*tau*d_exp_model/(1-gamma)
    diss_penalty = 1/(2*gamma*(1-gamma)) *(model_adv*(d_inf_policy/d_inf_model + d_exp_policy/d_exp_model) + 
                                           gamma**2*delta_U*d_exp_policy*d_inf_policy/(1-gamma))
    term_1 = model_adv/(gamma*delta_U*d_inf_model)
    term_2 = 1/(2*(1-gamma))* d_exp_model*(d_inf_policy/d_inf_model + d_exp_policy/d_exp_model)

    adv += term_1
    diss_penalty += term_2
   
    return adv - diss_penalty - bias




def compute_teleport_bound_1_tau_test(tau, policy_adv, model_adv, gamma, d_inf_model, 
                                 d_inf_policy, d_exp_model, d_exp_policy, delta_U):
    return (16*d_exp_model**2*d_inf_model*delta_U*gamma**4*tau - 16*d_exp_model**2*d_inf_model*delta_U*gamma**3*tau + d_exp_model**2*d_inf_policy**2*delta_U**2*gamma**2 + 4*d_exp_model**2*d_inf_policy*delta_U*gamma**3 - 4*d_exp_model**2*d_inf_policy*delta_U*gamma**2 + 4*d_exp_model**2*gamma**4 - 8*d_exp_model**2*gamma**3 + 4*d_exp_model**2*gamma**2 - 4*d_exp_model*d_exp_policy*d_inf_model*d_inf_policy*delta_U**2*gamma**3 + 2*d_exp_model*d_exp_policy*d_inf_model*d_inf_policy*delta_U**2*gamma**2 + 4*d_exp_model*d_exp_policy*d_inf_model*delta_U*gamma**3 - 4*d_exp_model*d_exp_policy*d_inf_model*delta_U*gamma**2 - 8*d_exp_model*d_inf_model*delta_U*gamma**3*policy_adv + 8*d_exp_model*d_inf_model*delta_U*gamma**2*policy_adv + 4*d_exp_model*d_inf_policy*delta_U*gamma**2*model_adv - 4*d_exp_model*d_inf_policy*delta_U*gamma*model_adv + 8*d_exp_model*gamma**3*model_adv - 16*d_exp_model*gamma**2*model_adv + 8*d_exp_model*gamma*model_adv + d_exp_policy**2*d_inf_model**2*delta_U**2*gamma**2 + 4*d_exp_policy*d_inf_model*delta_U*gamma**2*model_adv - 4*d_exp_policy*d_inf_model*delta_U*gamma*model_adv + 4*gamma**2*model_adv**2 - 8*gamma*model_adv**2 + 4*model_adv**2)/(8*d_exp_model*d_inf_model*delta_U*gamma**2*(gamma - 1)**2)


"""
    Compute the number of time steps to converge to the original model
"""
def compute_n(gamma, tau, eps_model):
    return np.ceil(2*gamma*tau/((1-gamma)*eps_model)) - 1

"""
    Compute the model step size to converge to the original model in n steps
"""
def compute_eps_model(gamma, tau, n):
    return 2*gamma*tau/((n+1)*(1-gamma))

def compute_tau_prime(gamma, tau, eps_model):
    tau_prime = tau - eps_model*(1 - gamma)/(2*gamma)
    tau_prime = max(0, tau_prime)
    return tau_prime


def stochastic_argmax(value_list):
    max_indices = np.where(value_list == np.max(value_list))[0]
    if len(max_indices) == 1:
        return max_indices[0]
    
    return np.random.choice(max_indices)