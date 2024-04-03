import numpy as np
from gymnasium import Env
from scipy.special import softmax

"""
    Sample from categorical distribution
    Args:
        - prob_n (np.ndarray) : probability distribution vector [size]
        - np_random: random number generator
    return (int): a categorical sampling from the given probability distribution
"""
def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    # Compute cumulative sum of the probability vector, that is used for the CDF of the categorical distribution
    csprob_n = np.cumsum(prob_n)
    # np_randoim.random() generates a random number in [0,1], then we find the first index of the cumulative sum that is greater than the random number
    return (csprob_n > np_random.random()).argmax()


"""
    Compute the average reward when picking action a in state s. It is evaluated as R(s,a) = sum_s' P(s'|s,a) * R(s,a,s')
    
    Args:    
        - P_mat (np.ndarray): probability transition function [size, nA, size] or state teleport probability distribution [size]
        - reward (np.ndarray): the reward function [size, nA, size]
    return (np.ndarray):the average reward when picking action a in state s as a as [ns, nA] matrix
"""
def compute_r_s_a(P_mat, reward):
    # Average reward when taking action a in state s, of size |S|x|A|
    size, nA, _ = reward.shape
    # If the probability transition function is the teleport probability vector, we simply replicate it to match the shape of the reward function
    if len(P_mat.shape) == 1:
        P_mat = np.tile(P_mat, (nA, size)).T
        P_mat = P_mat.reshape((size, nA, size))
        
    r_s_a =np.zeros(shape=(size, nA))
    for s in range(size):
        for a in range(nA):
            for s_prime in range(size):
                r_s_a[s,a] = r_s_a[s,a] + P_mat[s][a][s_prime] * reward[s, a, s_prime]
    return r_s_a

"""
    Compute the probability of moving from state s to state sprime, under policy pi. It is evaluated as P(s'|s) = sum_a pi(a|s) * P(s'|s,a)
    Args:
        - P_mat (np.ndarray): probability transition function [size, nA, size]
        - pi (np.ndarray): the given policy [size, nA]

    return (np.ndarray): the probability of moving from state s to state sprime under policy pi [size, size] 
"""
def compute_p_sprime_s(P_mat, pi):
    size, nA = pi.shape
    P_sprime_s = np.zeros((size, size), dtype='float64')
    for s in range(size):
        for a in range(nA):
            for s_prime in range(size):
                P_sprime_s[s][s_prime] += pi[s, a] * P_mat[s][a][s_prime]
    return P_sprime_s


"""
    Compute the discounted state distribution as d = (1-gamma) * mu * (I - gamma*P)^-1
    Args:
        - mu (np.ndarray): initial state distribution [size]
        - P_mat (np.ndarray): probability transition function of the original problem [size, nA, size]
        - xi (np.ndarray): state teleport probability distribution [size]
        - pi (np.ndarray): the given policy [size, nA]
        - gamma (float): discount factor
        - tau (float): teleport probability
    return (np.ndarray): the discount state distribution as a vector [size]
"""
def compute_d_from_tau(mu, P_mat, xi, pi, gamma, tau):
    size, nA = pi.shape
    I = np.eye(size)
    Xi = np.tile(xi.astype('float64'), size).reshape((size, size))
    P_sprime_s = compute_p_sprime_s(P_mat, pi)
    inv = np.linalg.inv(I - gamma*(1-tau)*P_sprime_s-tau*gamma*Xi)
    d = np.matmul((1-gamma)*mu, inv)
    return d


"""
    Compute the discounted state distribution as d = (1-gamma) * mu * (I - gamma*P)^-1
    Args:
        - mu (np.ndarray): initial state distribution [size]
        - P_mat (np.ndarray): probability transition function [size, nA, size]
        - pi (np.ndarray): the given policy [size, nA]
        - gamma (float): discount factor
    return (np.ndarray): the discount state distribution as a vector [size]
"""
def compute_d(mu, P_mat, pi, gamma):
    size, nA = pi.shape
    I = np.eye(size)
    P_sprime_s = compute_p_sprime_s(P_mat, pi)
    inv = np.linalg.inv(I - gamma*P_sprime_s)
    d = np.matmul((1-gamma)*mu, inv)
    return d


"""
"""
def compute_grad_d(P_mat, P_mat_tau, xi, mu, pi, gamma):
    size, nA = pi.shape
    I = np.eye(size)
    P_sprime_s = compute_p_sprime_s(P_mat, pi)
    P_sprime_s_tau = compute_p_sprime_s(P_mat_tau, pi)

    Xi = np.tile(xi, size).reshape((size, size))
    model_diff = Xi-P_sprime_s

    M = np.linalg.inv(I - gamma*P_sprime_s_tau)
    
    grad_d = np.matmul(M, model_diff)
    grad_d = np.matmul(grad_d, M)
    grad_d = np.matmul(mu, grad_d)
    grad_d = (1-gamma)*gamma*grad_d
    return grad_d


"""
    Compute the discounted state action distribution under policy pi as delta = pi * d
    Args:
        - d (np.ndarray) : the discounted state distribution as a vector [size]
        - pi (np.ndarray): the given policy [size, nA]    
    return (np.ndarray): the discount state action distribution under policy pi as [size, nA]
"""
def compute_delta(d, pi):
    delta = np.zeros(pi.shape)
    size, nA = pi.shape
    for s in range(size):
        for a in range(nA):
            delta[s,a] = pi[s, a] * d[s]
    return delta

"""
    Compute the discounted state action distribution under policy pi as delta = pi * d, computing d implicitly
    Args:
        - mu (np.ndarray): initial state distribution [size]
        - P_mat (np.ndarray): probability transition function [size, nA, size]
        - pi (np.ndarray): the given policy [size, nA]
        - gamma (float): discount factor
            
    return (np.ndarray): the discount state action distribution under policy pi as [size, nA]
"""
def get_delta(mu, P_mat, pi, gamma):
    d = compute_d(mu, P_mat, pi, gamma)
    return compute_delta(d, pi)

"""
    Extract the policy from a given state action value function
    Args:
        - Q (np.ndarray): the state action value function [size, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy

    return (np.ndarray): the greedy policy according to Q, as [size, nA]
"""
def get_policy(Q, det=True):
    pi = np.zeros(Q.shape)
    if det:
        for x in range(Q.shape[0]):
            pi[x,np.argmax(Q[x])] = 1
    else:
        for x in range(Q.shape[0]):
            #pi[x] = softmax(Q[x]) 
            pi[x] = Q[x]/np.sum(Q[x])
    return pi


"""
    Extract the value function from a given state action value function
    Args:
        - Q (np.ndarray): the state action value function [size, nA]
        - det (bool): deterministic flag. Whether or not considering a deterministic policy
    return (np.ndarray): the value function as V(s) = sum_a Q(s,a) * pi(a|s) [size]
"""
def get_value_function(Q, det=True):
    pi = get_policy(Q, det)
    V = np.zeros(Q.shape[0])
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            V[s] = V[s] + Q[s,a]*pi[s,a]
    return V

"""
    Compute the expected discounted sum of returns as J = 1/(1-gamma) sum_s d(s) * sum_a pi(a|s) * R(s,a)
    Args:    
        - r_s_a (np.ndarray): the average reward when picking action a in state s [size, nA]
        - pi (np.ndarray): the given policy [size, nA]
        - d (np.ndarray): the discounted state distribution as a vector [size]
        - gamma (float): discount factor
    return (float): the expected discounted sum of returns as a scalar value
"""
def compute_j(r_s_a, pi, d, gamma):
    size, nA = pi.shape
    J = 0

    for s in range(size):
        sum = 0
        for a in range(nA):
            sum = sum + r_s_a[s, a]*pi[s,a]
        J = J + d[s]*sum
    J = J/(1-gamma)
    return J

"""
    Utility function to get the expected discounted sum of returns computing d and r_s_a implicitly
    Args:    
        - P_mat (np.ndarray): probability transition function [size, nA, size]
        - pi (np.ndarray): the given policy [size, nA]
        - reward (np.ndarray): the reward function [size, nA, size]
        - gamma (float): discount factor
        - mu (np.ndarray): initial state distribution [size]

    return (float): the expected discounted sum of returns as a scalar value
"""
def get_expected_avg_reward(P_mat, pi, reward, gamma, mu):
    r_s_a = compute_r_s_a(P_mat, reward)
    d = compute_d(mu, P_mat, pi, gamma)
    return compute_j(r_s_a, pi, d, gamma)

"""
    Compute the state action nextstate value function U(s, a, s') = R(s,a,s') + gamma * V(s')
    Args:
        - r_s_a (np.ndarray): the average reward when picking action a in state s [size, nA]
        - gamma (float): discount factor
        - V (np.ndarray): the value function [size]
    return (np.ndarray): the state action nextstate value function [size, nA, size]
"""
def compute_state_action_nextstate_value_function(r_s_a, gamma, V):
    size, nA = r_s_a.shape
    U = np.zeros((size, nA, size))
    for s in range(size):
        for a in range(nA):
            U[s, a] = r_s_a[s, a] + gamma*V
    return U

"""
    Utility function to get the state action nextstate value function U(s, a, s') = R(s,a,s') + gamma * V(s')
    Args:
        - P_mat (np.ndarray): probability transition function [size, nA, size]
        - reward (np.ndarray): the reward function [size, nA, size]
        - gamma (float): discount factor
        - Q (np.ndarray): the state action value function [size, nA]
        - det (bool): deterministic flag. Whether or not considering a deterministic policy
    return (np.ndarray): the state action nextstate value function [size, nA, size]
"""
def get_state_action_nextstate_value_function(P_mat, reward, gamma, Q, det=True):
    r_s_a = compute_r_s_a(P_mat, reward)
    V = get_value_function(Q, det)
    return compute_state_action_nextstate_value_function(r_s_a, gamma, V)

"""
    Extract the state action value function from a given state action nextstate value function Q = sum_s' P(s'|s,a) * U(s,a,s')
    Args:
        - P_mat (np.ndarray): probability transition function [size, nA, size]
        - U (np.ndarray): state action next state value function as[size, nA, size]
    return (np.ndarray): the state action value function [size, nA] 
"""
def rebuild_Q_from_U(P_mat, U):
    size, nA, _ = U.shape
    Q = np.zeros((size, nA))
    
    for s in range(size):
        for a in range(nA):
            for s_prime in range(size):
                Q[s, a] = Q[s,a] +  U[s, a, s_prime] * P_mat[s][a][s_prime]
    
    return Q

"""
    Compute the state policy advantage function A(s,a) = Q(s,a) - V(s)
    Args:
        - Q (np.ndarray): the state action value function [size, nA]
        - V (np.ndarray): the value function [size]
    return (np.ndarray): the policy advantage function [size, nA]
"""
def compute_policy_advantage_function(Q, V):
    size, nA = Q.shape
    pol_adv = np.zeros((size, nA))
    for s in range(size):
        for a in range(nA):
            pol_adv[s, a] = Q[s,a] - V[s]
    return pol_adv

"""
    Utility function to get the policy advantage function extracting V implicitly. A = Q - V
    Args:
        - Q (np.ndarray): the state action value function [size, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the policy advantage function [size, nA]
"""
def get_policy_advantage_function(Q, det=True):
    V = get_value_function(Q, det)
    return compute_policy_advantage_function(Q, V)

"""
    Compute the model advantage function A(s, a, s') = U(s, a, s') - Q(s, a)
    Args:
        - U (np.ndarray): the state action nextstate value function [size, nA, size]
        - Q (np.ndarray): the state action value function [size, nA]
    return (np.ndarray): the model advantage function [size, nA, size]
"""
def compute_model_advantage_function(U, Q):
    size, nA = Q.shape
    model_adv = np.zeros((size, nA, size))

    for s in range(size):
        for a in range(nA):
            for s_prime in range(size):
                model_adv[s, a, s_prime] = U[s, a, s_prime] - Q[s, a]
    return model_adv

"""
    Utility function to get the model advantage function computing U implicitly. A = U - Q
    Args:
        - P_mat (np.ndarray): probability transition function [size, nA, size]
        - reward (np.ndarray): the reward function [size, nA, size]
        - gamma (float): discount factor
        - Q (np.ndarray): the state action value function [size, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the model advantage function [size, nA, size]
"""
def get_model_advantage_function(P_mat, reward, gamma, Q, det=True):
    U = get_state_action_nextstate_value_function(P_mat, reward, gamma, Q, det)
    return compute_model_advantage_function(U, Q)


"""
    Compute the relative policy advantage function A_pi_pi_prime(s)
    Args:
        - pi_prime (np.ndarray): the new policy to be compared [size, nA]
        - A (np.ndarray): the policy advantage function [size, nA]
    return (np.ndarray): the relative policy advantage function [size]
"""
def compute_relative_policy_advantage_function(pi_prime, A):
    size, nA = pi_prime.shape
    rel_pol_adv = np.zeros(size)
    for s in range(size):
        for a in range(nA):
            rel_pol_adv[s] = rel_pol_adv[s] + pi_prime[s, a]*A[s, a]
    return rel_pol_adv


"""
    Utility function to get the relative policy advantage function A_pi_pi_prime(s) = sum_a pi_prime(a|s) * A(s,a)
    Args:
        - Q (np.ndarray): the state action value function [size, nA]
        - pi_prime (np.ndarray): the new policy to be compared [size, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the relative policy advantage function [size]
"""
def get_relative_policy_advantage_function(Q, pi_prime, det=True):
    A = get_policy_advantage_function(Q, det)
    return compute_relative_policy_advantage_function(pi_prime, A)

"""
    Utility function to get the relative policy advantage function A_pi_pi_prime(s) = sum_a (pi_prime(a|s) - pi(a|s)) * Q(s,a).
    Same result of get_relative_policy_advantage_function, but emphasizing the difference among the two policies.
    Args:
        - Q (np.ndarray): the state action value function [size, nA]
        - pi_prime (np.ndarray): the new policy to be compared [size, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the relative policy advantage function [size]
"""
def get_relative_policy_advantage_function_from_delta_policy(Q, pi_prime, det=True):
    size, nA = pi_prime.shape
    rel_pol_adv = np.zeros(size)
    pi = get_policy(Q, det)
    for s in range(size):
        for a in range(nA):
            rel_pol_adv[s] += (pi_prime[s, a] - pi[s,a]) * Q[s,a]
    return rel_pol_adv



"""
    Compute the relative model advantage function A_tau_tauprime(s,a) = sum_s' P_tau_prime(s'|s,a) * A(s,a,s')
    Args:
        - P_mat_prime (nd.array): the probability transition function to be compared [size, nA, size]
        - A (np.ndarray): the model advantage function [size, nA]
    return (np.ndarray): the relative model advantage function [size, nA]
"""
def compute_relative_model_advantage_function(P_mat_prime, A):
    size, nA, _ = A.shape
    rel_model_adv = np.zeros((size, nA))
    for s in range(size):
        for a in range(nA):
            rel_model_adv[s, a] = np.matmul(P_mat_prime[s][a], np.transpose(A[s, a]))
    return rel_model_adv

"""
    Compute the relative model advantage function hat. \hat{A}_tau(s,a) = sum_s' (P_tau(s'|s,a) - xi(s')) * U(s,a,s')
    N.B. to get the actual relative model advantage function you have to multiply times (\tau - \tau')
    Args:
        - P_mat (np.ndarray): probability transition function of the original problem, with tau=0 [size, nA, size]
        - xi (np.ndarray): state teleport probability distribution [size]
        - U (np.ndarray): state action next state value function [size, nA, size]
    return (np.ndarray): the relative model advantage function hat [size, nA]
"""
def compute_relative_model_advantage_function_hat(P_mat, xi, U):
    size, nA, _ = U.shape
    model_adv_hat = np.zeros((size, nA))
    for s in range(size):
        for a in range(nA):
            for s1 in range(size):
                model_adv_hat[s, a] = model_adv_hat[s,a] + (P_mat [s][a][s1]- xi[s1])*U[s,a,s1]
    return model_adv_hat

"""
    Compute the relative model advantage function A_tau_tau_prime(s,a) = (tau - tau') * \hat{A}_tau(s,a).
    Args:
        - P_mat (np.ndarray): probability transition function of the original problem, with tau=0 [size, nA, size]
        - xi (np.ndarray): state teleport probability distribution [size]
        - U (np.ndarray): state action next state value function [size, nA, size]
        - tau (float): teleport probability
        - tau_prime (float): new teleport probability
    return (np.ndarray): the relative model advantage function [size, nA]
"""
def compute_relative_model_advantage_function_from_delta_tau(P_mat, xi, U, tau, tau_prime):
    model_adv_hat = compute_relative_model_advantage_function_hat(P_mat, xi, U)
    return (tau - tau_prime)*model_adv_hat

"""
    Compute the discounted distribution relative model advantage function \mathcal{A}_tau_tauprime = sum_s sum_a delta(s,a) * A_tau_tauprime(s,a)
    Args:
        - A (np.ndarray): the model advantage function [size, nA]
        - delta (np.ndarray): the discount state action distribution under policy pi [size, nA]
    return (float): the discounted distribution relative model advantage function as a scalar
"""
def compute_discounted_distribution_relative_model_advantage_function(A, delta):
    dis_rel_model_adv = 0
    size, nA = delta.shape
    for s in range(size):
        for a in range(nA):
            dis_rel_model_adv = dis_rel_model_adv + A[s, a]* delta[s, a]
    return dis_rel_model_adv

"""
    Compute the discounted distribution relative model advantage function hat \mathcal{A}_tau_tauprime = sum_s sum_a delta(s,a) * \hat{A}_tau(s,a)
    N.B. to get the actual discounted distribution relative model advantage function you have to multiply times (\tau - \tau')
    Args:
        - A_tau_hat (np.ndarray): the relative model advantage function hat [size, nA]
        - delta (np.ndarray): the discount state action distribution under policy pi [size, nA]
    return (float): the discounted distribution relative model advantage function as a scalar
"""
def compute_discounted_distribution_relative_model_advantage_function_hat(A_tau_hat, delta):
    dis_rel_model_adv = 0
    size, nA = A_tau_hat.shape
    for s in range(size):
        for a in range(nA):
            dis_rel_model_adv = dis_rel_model_adv + A_tau_hat[s, a]*delta[s, a]
    return dis_rel_model_adv

"""
    Compute the discounted relative model advantage function dis_model_adv(s,a) = (tau - tau') * \hat{A}_tau(s,a).
    Args:
        - P_mat (np.ndarray): probability transition function of the original problem, with tau=0 [size, nA, size]
        - xi (np.ndarray): state teleport probability distribution [size]
        - U (np.ndarray): state action next state value function [size, nA, size]
        - tau (float): teleport probability
        - tau_prime (float): new teleport probability
    return (np.ndarray): the relative model advantage function [size, nA]
"""
def compute_discounted_distribution_relative_model_advantage_function_from_delta_tau(A_tau_hat, delta, tau, tau_prime):
    dis_model_adv_hat = compute_discounted_distribution_relative_model_advantage_function_hat(A_tau_hat, delta)
    return (tau - tau_prime)*dis_model_adv_hat

# Here

""" 
    Compute the expected value of the difference among two probability distribution over (s,a)~delta.
    It works in two ways:
        - if P_mat_tau and P_mat_tauprime have the same shape, it computes the expected difference among the two probability transition functions
        - if P_mat_tauprime has the same shape of P_mat_tau.shape[0], meaning that we are comparing P_mat_tau with a probability vector, it computes the expected value of the difference among each row of P_mat_tau and the probability vector P_mat_tauprime
    Args:
        - P_mat_tau (np.ndarray): probability transition function [size, nA, size]
        - P_mat_tauprime (np.ndarray): probability transition function [size, nA, size] or probability vector [size]
        - delta (np.ndarray): the discount state action distribution under policy pi [size, nA]
    return (float): the expected difference among the two probability transition functions
"""
def get_expected_difference_transition_models(P_mat_tau, P_mat_tauprime, delta):
    de = 0
    size, nA, _ = P_mat_tau.shape
    if P_mat_tau.shape == P_mat_tauprime.shape:
        for s in range(size):
            for a in range(nA):
                de += np.linalg.norm(P_mat_tauprime[s, a] - P_mat_tau[s, a], ord=1) * delta[s,a]
    elif P_mat_tau.shape[0] == P_mat_tauprime.shape[0]:
         for s in range(size):
            for a in range(nA):
                de += np.linalg.norm(P_mat_tau[s,a] - P_mat_tauprime, ord=1) * delta[s,a]
    else:
        raise ValueError('P_mat_tau and P_mat_tauprime have incompatible shapes')
    return de

"""
    Compute the superior difference among two probability transition functions.
    It works in two ways:
        - if P_mat_tau and P_mat_tauprime have the same shape, it computes the superior difference among the two probability transition functions
        - if P_mat_tauprime has the same shape of P_mat_tau.shape[0], meaning that we are comparing P_mat_tau with a probability vector, it computes the superior difference among each row of P_mat_tau and the probability vector P_mat_tauprime
    Args:
        - P_mat_tau (np.ndarray): probability transition function [size, nA, size]
        - P_mat_tauprime (np.ndarray): probability transition function [size, nA, size] or probability vector [size]
    return (float): the superior difference among the two probability transition functions
"""
def get_sup_difference_transition_models(P_mat_tau, P_mat_tauprime):
    size, nA, _ = P_mat_tau.shape

    # Here we are considering two different probability transition functions, with the same shape. Nominally, P_tau and P_tau_prime
    if P_mat_tau.shape == P_mat_tauprime.shape:
        return np.max(np.abs(P_mat_tau - P_mat_tauprime))
    
    # Here we are considering the case in which we ar compering the probability transition function of the original problem, with the state teleport probability distribution xi
    elif P_mat_tau.shape[0] == P_mat_tauprime.shape[0]:
        Xi = np.tile(P_mat_tauprime, (nA, size)).T
        Xi = Xi.reshape((size, nA, size))
        return np.max(np.abs(P_mat_tau - Xi))
    else:
        raise ValueError('P_mat_tau and P_mat_tauprime have incompatible shapes')
    return de

"""
    Compute the superior difference between any two elements of the state action value function Q
    Args:
        - Q (np.ndarray): the state action value function [size, nA]
    return (float): the superior difference between any two elements of the state action value function Q
"""
def get_sup_difference_q(Q):
    size, nA = Q.shape
    sup = -np.inf
    for s in range(size):
        for a in range(nA):
            for s1 in range(size):
                for a1 in range(nA):
                    diff = np.abs(Q[s,a]-Q[s1,a1])
                    if diff > sup:
                        sup = diff
    return sup
    

"""
    Compute the performance improvement lower bound as lb = dis_rel_model_adv/(1-gamma) - 2*gamma^2*(tau-tauprime)^2*de*dq*dinf/(2*(1-gamma)^2) 
    Where:
        - dis_rel_model_adv = sum_{s,a} delta(s,a) * rel_model_adv(s,a)
        - de = \sum_{s,a} delta(s,a) * ||P_tau_prime(s,a) - P_tau(s,a)||_1
        - dq = sup_{s,a,s', a'} |Q_tau(s,a) - Q_tau(s',a')|
        - dinf = sup_{s,a} ||P_tau_prime(s,a) - P_tau(s,a)||_1
    Args:
        - P_mat_tau (np.ndarray): probability transition function [size, nA, size]
        - P_mat_tauprime (np.ndarray): new probability transition function [size, nA, size]
        - reward (np.ndarray): the reward function [size, nA, size]
        - gamma (float): discount factor
        - tau (float): teleport probability
        - tau_prime (float): new teleport probability
        - Q (np.ndarray): the state action value function [size, nA]
        - mu (np.ndarray): initial state distribution [size]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (float): the performance improvement lower bound    
"""
def get_performance_improvement_lower_bound(P_mat_tau, P_mat_tauprime , reward, gamma, tau, tau_prime, Q, mu, det=True):
    # Compute the advantage
    pi = get_policy(Q, det)
    delta = get_delta(mu, P_mat_tau, pi, gamma)

    model_adv = get_model_advantage_function(P_mat_tau, reward, gamma, Q, det)
    rel_adv = compute_relative_model_advantage_function(P_mat_tauprime, model_adv)
    dis_rel_model_adv = compute_discounted_distribution_relative_model_advantage_function(rel_adv, delta)
    adv = dis_rel_model_adv/(1-gamma)

    # Compute the dissimilarity penalization
    dq = get_sup_difference_q(Q)
    de = get_expected_difference_transition_models(P_mat_tau, P_mat_tauprime, delta)
    dinf = get_sup_difference_transition_models(P_mat_tau, P_mat_tauprime)
    diss_pen = gamma**2*(tau - tau_prime)**2*dq*de*dinf/(2*(1-gamma)**2)

    return adv - diss_pen


"""
    Compute the optimal teleport probability tau_prime that maximizes the performance improvement lower bound
    Args:
        - P_mat_tau (np.ndarray): probability transition function [size, nA, size]
        - reward (np.ndarray): the reward function [size, nA, size]
        - gamma (float): discount factor
        - tau (float): teleport probability
        - Q (np.ndarray): the state action value function [size, nA]
        - mu (np.ndarray): initial state distribution [size]
        - xi (np.ndarray): state teleport probability distribution [size]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (float): the optimal teleport probability tau_prime that maximizes the performance improvement lower bound
"""
def  get_optimal_tauprime(P_mat_tau, reward, gamma, tau, Q, mu, xi, det=True):

    # Compute the discounted relative model advantage hat

    U = get_state_action_nextstate_value_function(P_mat_tau, reward, gamma, Q, det)
    rel_model_adv_hat = compute_relative_model_advantage_function_hat(P_mat_tau, xi, U)
    pi = get_policy(Q, det)
    delta = get_delta(mu, P_mat_tau, pi, gamma)
    dis_rel_model_adv_hat = compute_discounted_distribution_relative_model_advantage_function_hat(rel_model_adv_hat, delta)

    dq = get_sup_difference_q(Q)
    de = get_expected_difference_transition_models(P_mat_tau, xi, delta)
    dinf = get_sup_difference_transition_models(P_mat_tau, xi)

    return tau - dis_rel_model_adv_hat*(1-gamma)/(2*gamma**2*dq*de*dinf)


"""
    Extract the Q_hat state action function from a given state action value function Q
    and the probability transition function.
    Args:
        - P_mat (np.ndarray): probability transition function [size, nA, size] or state teleport probability distribution [size]
        - r_s_a (np.ndarray): the average reward when picking action a in state s [size, nA]
        - gamma (float): discount factor
        - Q (np.ndarray): the state action value function [size, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the Q_hat state action value function [size, nA]
"""
def get_q_hat(P_mat, r_s_a, gamma, Q, det=True):
    ns, nA = Q.shape
    q_hat = np.zeros((ns, nA))

    # If P_mat is the state teleport probability vector, we simply replicate it to match the shape of the reward function
    if len(P_mat.shape) == 1:
        P_mat = np.tile(P_mat, (nA, ns)).T
        P_mat = P_mat.reshape((ns, nA, ns))
    
    # Extract the policy from Q
    pi = get_policy(Q, det=det)
    # weight the state action value function by the given policy, main diagonal of the matrix product among pi and Q^T
    # Can be pre-computed for computational efficiency
    pi_Q = np.einsum('ij,ij->i', pi, Q)
    
    for s in range(ns):
        for a in range(nA):
            q_hat[s,a] = r_s_a[s,a] + gamma * np.matmul(P_mat[s,a,:], pi_Q)
    return q_hat


"""
    Compute the gradient of the expected discounted sum of returns as grad_J = 1/(1-gamma)sum_s d(s) * sum_a pi(a|s) * (Q_xi(s,a) - Q_p(s,a))
    Args:
        - pi (np.ndarray): the given policy [size, nA]
        - Q_p (np.ndarray): the state action value function under policy p [size, nA]
        - Q_xi (np.ndarray): the state action value function under policy xi [size, nA]
        - d (np.ndarray): the discounted state distribution as a vector [size]
        - gamma (float): discount factor
"""
def compute_grad_j(pi, Q_p, Q_xi, d, gamma):
    size, nA = pi.shape
    grad = 0
    for s in range(size):
        sum = 0
        for a in range(nA):
            sum += pi[s,a]*(Q_xi[s,a] - Q_p[s,a])
        grad += d[s]*sum
    grad = grad/(1-gamma)
    return grad

