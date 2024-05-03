import numpy as np
from gymnasium import Env
from scipy.special import softmax
import math

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
def compute_transition_kernel(P_mat, pi):
    nS, nA = pi.shape
    P_sprime_s = np.zeros((nS, nS), dtype='float64')
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                P_sprime_s[s][s_prime] += pi[s, a] * P_mat[s][a][s_prime]
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
    V = np.zeros(Q.shape[0])
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            V[s] += Q[s,a]*pi[s,a]
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
    nS, nA, _ = reward.shape
    U = np.zeros((nS, nA, nS))
    for s in range(nS):
        for a in range(nA):
            U[s, a] = reward[s, a] + gamma*V
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
def compute_relative_policy_advantage_function(pi_prime, pi, Q):
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
def compute_d_from_tau(mu, P_mat, xi, pi, gamma, tau):
    nS, nA = pi.shape
    I = np.eye(nS)
    Xi = np.tile(xi.astype('float64'), nS).reshape((nS, nS))
    P_sprime_s = compute_transition_kernel(P_mat, pi)
    inv = np.linalg.inv(I - gamma*(1-tau)*P_sprime_s-tau*gamma*Xi)
    d = np.matmul((1-gamma)*mu, inv)
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
    P_sprime_s = compute_transition_kernel(P_mat, pi)

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
    delta = np.zeros(pi.shape)
    nS, nA = pi.shape
    for s in range(nS):
        for a in range(nA):
            delta[s,a] = pi[s, a] * d[s]
    return delta

######################### Difference Metrics #########################
"""
    Compute the superior difference between any two elements of the Q function
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
    return (float): the superior difference between any two elements of the state action value function Q
"""
def get_sup_difference_Q(Q):
    return np.max(Q) - np.min(Q)

"""
    Compute the superior difference between any two elements of the U function
    Args:
        - U (np.ndarray): the state action next state value function [nS, nA, nS]
    return (float): the superior difference between any two elements of the state action next state value function U
"""
def get_sup_difference_U(U):
    return np.max(U) - np.min(U)


"""
    Compute the superior of the l1 norm of the difference between two policies
    Args:
        - pi (np.ndarray): the policy [nS, nA]
        - pi_prime (np.ndarray): the new policy [nS, nA]
    return (float): the superior of the l1 norm of the difference between two policies
"""
def get_d_inf_policy(pi, pi_prime):
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
    adv = (alpha * policy_adv + abs(tau- tau_prime) * model_adv)/(1-gamma)
    bias = 0
    if biased:
        bias = gamma*tau*d_exp_model/(1-gamma)
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
    sign = - model_adv
    adv = (1-gamma)*(model_adv)/(gamma**2*delta_U*d_exp_model*d_inf_model)
    bias = 0
    if biased:
        bias = (1-gamma)*(gamma*d_exp_model)/(gamma**2*delta_U*d_exp_model*d_inf_model)
    tau_prime = 0
    if(sign <= 0):
        tau_prime = tau - adv -bias
    else:
        tau_prime = tau + adv - bias
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
    sign = 1/(2*gamma) * (d_exp_policy/d_exp_model + d_inf_policy/d_inf_model) - (1-gamma)*(model_adv+gamma*d_exp_model)/(gamma**2*delta_U*d_exp_model*d_inf_model)
    tau_prime_0 = 0
    adv = (1-gamma)*(model_adv)/(gamma**2*delta_U*d_exp_model*d_inf_model)
    bias = 0
    if biased:
        bias = (1-gamma)*(gamma*d_exp_model)/(gamma**2*delta_U*d_exp_model*d_inf_model)
    diss = 1/(2*gamma) * (d_exp_policy/d_exp_model + d_inf_policy/d_inf_model)
    if sign <= 0: #Tau_prime is smaller than tau
        tau_prime_0 = tau - adv + diss -bias
    else:
        tau_prime_0 = tau + adv - diss - bias 
    return round(tau_prime_0, 5)

def get_teleport_bound_optimal_values(pol_adv, model_adv, delta_U, d_inf_pol, d_exp_pol,
                                       d_inf_model, d_exp_model, tau, gamma, biased=True):
    # refuse tau' if tau' < tau
    optimal_values = []
    if d_inf_pol != 0 and d_exp_pol != 0:
        # optimal value for alpha with tau'=tau
        alpha_tau = compute_alpha_tau(pol_adv, gamma, delta_U, d_exp_pol, d_inf_pol)
        if math.isnan(alpha_tau):
            print("pol_adv: {}, gamma: {}, delta_U: {}, d_exp_pol: {}, d_inf_pol: {}".format(pol_adv, gamma, delta_U, d_exp_pol, d_inf_pol))
        if alpha_tau >= 0 and alpha_tau <= 1:
            optimal_values.append((round(alpha_tau, 5), round(tau, 5)))
        """elif alpha_tau < 0:
            optimal_values.append((0, tau))
        else:
            optimal_values.append((1, tau))"""
        
        # optimal value for alpha with tau'=0
        alpha_0 = compute_alpha_0(pol_adv, tau, gamma, delta_U, d_exp_pol, d_inf_pol, d_exp_model, d_inf_model)
        if math.isnan(alpha_0):
            print("pol_adv: {}, gamma: {}, delta_U: {}, d_exp_pol: {}, d_inf_pol: {}, d_exp_model: {}, d_inf_model: {}".format(pol_adv, gamma, delta_U, d_exp_pol, d_inf_pol, d_exp_model, d_inf_model))
        if alpha_0 >= 0 and alpha_0 <= 1:
            optimal_values.append((round(alpha_0,5), 0.))
        """elif alpha_0 < 0:
            optimal_values.append((0, 0))
        else:
            optimal_values.append((1, 0))"""
    
    # optimal value for tau' with alpha=0
    tau_prime_0 = compute_tau_prime_0(tau, model_adv, gamma, d_exp_model, delta_U, d_inf_model, biased=biased)
    if math.isnan(tau_prime_0):
        print("model_adv: {}, gamma: {}, delta_U: {}, d_exp_model: {}, d_inf_model: {}".format(model_adv, gamma, delta_U, d_exp_model, d_inf_model))
    if tau_prime_0 >= 0 and tau_prime_0 <= 1:
        optimal_values.append((0., round(tau_prime_0,5)))
    elif tau_prime_0 < 0:
        optimal_values.append((0., 0.))
    else:
        print("Not valid tau_prime_0: {}".format(tau_prime_0))

    # optimal value for tau' with alpha=1
    tau_prime_1 = compute_tau_prime_1(tau, model_adv, gamma, d_exp_model, delta_U, d_inf_model, d_inf_pol, d_exp_pol, biased=biased)     
    if math.isnan(tau_prime_1):
        print("model_adv: {}, gamma: {}, delta_U: {}, d_exp_model: {}, d_inf_model: {}, d_inf_pol: {}, d_exp_pol: {}".format(model_adv, gamma, delta_U, d_exp_model, d_inf_model, d_inf_pol, d_exp_pol))
    if tau_prime_1 >= 0 and tau_prime_1 <= 1:
        optimal_values.append((1., round(tau_prime_1,5)))
    """elif tau_prime_1 < 0:
        optimal_values.append((1, 0))
    else:
        optimal_values.append((1, tau))"""

    return optimal_values

def get_teleport_bound_optima_pair(optimal_pairs, teleport_bounds):
    alpha_star, tau_star = optimal_pairs[np.argmax(teleport_bounds)]
    if (0, tau_star) and (1, tau_star) in optimal_pairs:
        alpha_star = 1
    if tau_star < 1e-3:
        tau_star = 0
    return (alpha_star, tau_star)


def compute_teleport_bound_alpha_tau(tau, policy_adv, 
                           gamma, d_inf_policy, 
                           d_exp_policy, d_exp_model, delta_U, biased=True
                           ):
    adv = policy_adv**2/(2*gamma*delta_U*d_exp_policy*d_inf_policy)
    bias = 0
    if biased:
        bias = gamma*tau*d_exp_model/(1-gamma)
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
def compute_teleport_bound_0_tau(tau, model_adv, gamma, d_inf_model, d_exp_model, delta_U):
    adv = model_adv**2/(2*gamma**2 *delta_U*d_exp_model*d_inf_model) + model_adv/(gamma*delta_U*d_inf_model) + d_exp_model/(2*delta_U*d_inf_model)
    bias = 2*gamma*tau*d_exp_model/(1-gamma)
    return adv-bias


"""
    Compute the teleport bound B(1, tau) associated to the pair (alpha=1, tau_prime=tau)
"""
def compute_teleport_bound_1_tau(tau, policy_adv, model_adv, gamma, d_inf_model, 
                                 d_inf_policy, d_exp_model, d_exp_policy, delta_U):
    sign = 1/(2*gamma) * (d_exp_policy/d_exp_model + d_inf_policy/d_inf_model) - (1-gamma)*(model_adv+gamma*d_exp_model)/(gamma**2*delta_U*d_exp_model*d_inf_model)
    adv = model_adv**2/(2*gamma**2 *delta_U*d_exp_model*d_inf_model) + d_exp_model/(2*delta_U*d_inf_model)
    adv += policy_adv/(1-gamma) + delta_U/(8*(1-gamma)**2*d_inf_model*d_exp_model)*(d_inf_model*d_exp_policy + d_inf_policy*d_exp_model)**2
    bias = 2*gamma*tau*d_exp_model/(1-gamma)
    diss_penalty = 1/(2*gamma*(1-gamma)) *(model_adv*(d_inf_policy/d_inf_model + d_exp_policy/d_exp_model) + 
                                           gamma**2*delta_U*d_exp_policy*d_inf_policy/(1-gamma))
    term_1 = model_adv/(gamma*delta_U*d_inf_model)
    term_2 = 1/(2*(1-gamma))* d_exp_model*(d_inf_policy/d_inf_model + d_exp_policy/d_exp_model)
    
    if sign <= 0: # tau_prime is smaller than tau
        adv += term_1
        diss_penalty += term_2
    else:
        adv -= term_1
        diss_penalty -= term_2

    return adv - diss_penalty - bias




def compute_teleport_bound_1_tau_test(tau, policy_adv, model_adv, gamma, d_inf_model, 
                                 d_inf_policy, d_exp_model, d_exp_policy, delta_U):
    return (16*d_exp_model**2*d_inf_model*delta_U*gamma**4*tau - 16*d_exp_model**2*d_inf_model*delta_U*gamma**3*tau + d_exp_model**2*d_inf_policy**2*delta_U**2*gamma**2 + 4*d_exp_model**2*d_inf_policy*delta_U*gamma**3 - 4*d_exp_model**2*d_inf_policy*delta_U*gamma**2 + 4*d_exp_model**2*gamma**4 - 8*d_exp_model**2*gamma**3 + 4*d_exp_model**2*gamma**2 - 4*d_exp_model*d_exp_policy*d_inf_model*d_inf_policy*delta_U**2*gamma**3 + 2*d_exp_model*d_exp_policy*d_inf_model*d_inf_policy*delta_U**2*gamma**2 + 4*d_exp_model*d_exp_policy*d_inf_model*delta_U*gamma**3 - 4*d_exp_model*d_exp_policy*d_inf_model*delta_U*gamma**2 - 8*d_exp_model*d_inf_model*delta_U*gamma**3*policy_adv + 8*d_exp_model*d_inf_model*delta_U*gamma**2*policy_adv + 4*d_exp_model*d_inf_policy*delta_U*gamma**2*model_adv - 4*d_exp_model*d_inf_policy*delta_U*gamma*model_adv + 8*d_exp_model*gamma**3*model_adv - 16*d_exp_model*gamma**2*model_adv + 8*d_exp_model*gamma*model_adv + d_exp_policy**2*d_inf_model**2*delta_U**2*gamma**2 + 4*d_exp_policy*d_inf_model*delta_U*gamma**2*model_adv - 4*d_exp_policy*d_inf_model*delta_U*gamma*model_adv + 4*gamma**2*model_adv**2 - 8*gamma*model_adv**2 + 4*model_adv**2)/(8*d_exp_model*d_inf_model*delta_U*gamma**2*(gamma - 1)**2)

