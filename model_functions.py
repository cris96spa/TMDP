import numpy as np
from gym import Env
from scipy.special import softmax

"""
    Sample from categorical distribution
    Args:
        - prob_n (np.ndarray) : probability distribution vector [nS]
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
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - reward (np.ndarray): the reward function [nS, nA, nS]
    return (np.ndarray):the average reward when picking action a in state s as a as [ns, nA] matrix
"""
def compute_r_s_a(P_mat, reward):
    # Average reward when taking action a in state s, of size |S|x|A|
    nS, nA, _ = P_mat.shape
    r_s_a =np.zeros(shape=(nS, nA))
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                r_s_a[s,a] = r_s_a[s,a] + P_mat[s][a][s_prime] * reward[s, a, s_prime]
            #print("Updating state {}, action {}, with {}".format(s, a, avg_rew))
    return r_s_a

"""
    Compute the probability of moving from state s to state sprime, under policy pi. It is evaluated as P(s'|s) = sum_a pi(a|s) * P(s'|s,a)
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - pi (np.ndarray): the given policy [nS, nA]

    return (np.ndarray): the probability of moving from state s to state sprime under policy pi [nS, nS] 
"""
def compute_p_sprime_s(P_mat, pi):
    nS, nA = pi.shape
    P_sprime_s = np.zeros((nS, nS))
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                P_sprime_s[s][s_prime] = P_sprime_s[s][s_prime] + pi[s, a] * P_mat[s][a][s_prime]
    return P_sprime_s


"""
    Compute the discounted state distribution as d = (1-gamma) * mu * (I - gamma*P)^-1
    Args:
        - mu (np.ndarray): initial state distribution [nS]
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - pi (np.ndarray): the given policy [nS, nA]
        - gamma (float): discount factor
    return (np.ndarray): the discount state distribution as a vector [nS]
"""
def compute_d(mu, P_mat, pi, gamma):
    nS, nA = pi.shape
    I = np.eye(nS)
    P_sprime_s = compute_p_sprime_s(P_mat, pi)
    inv = np.linalg.inv(I - gamma*P_sprime_s)
    d = np.matmul((1-gamma)*mu, inv)
    return d

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

"""
    Compute the discounted state action distribution under policy pi as delta = pi * d, computing d implicitly
    Args:
        - mu (np.ndarray): initial state distribution [nS]
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - pi (np.ndarray): the given policy [nS, nA]
        - gamma (float): discount factor
            
    return (np.ndarray): the discount state action distribution under policy pi as [nS, nA]
"""
def get_delta(mu, P_mat, pi, gamma):
    d = compute_d(mu, P_mat, pi, gamma)
    return compute_delta(d, pi)

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
            #pi[x] = softmax(Q[x]) 
            pi[x] = Q[x]/np.sum(Q[x])
    return pi


"""
    Extract the value function from a given state action value function
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - det (bool): deterministic flag. Whether or not considering a deterministic policy
    return (np.ndarray): the value function as V(s) = sum_a Q(s,a) * pi(a|s) [nS]
"""
def get_value_function(Q, det=True):
    pi = get_policy(Q, det)
    V = np.zeros(Q.shape[0])
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            V[s] = V[s] + Q[s,a]*pi[s,a]
    return V

"""
    Compute the expected discounted sum of returns as J = sum_s d(s) * sum_a pi(a|s) * R(s,a)
    Args:    
        - r_s_a (np.ndarray): the average reward when picking action a in state s [nS, nA]
        - pi (np.ndarray): the given policy [nS, nA]
        - d (np.ndarray): the discounted state distribution as a vector [nS]
        - gamma (float): discount factor
    return (float): the expected discounted sum of returns as a scalar value
"""
def compute_j(r_s_a, pi, d, gamma):
    nS, nA = pi.shape
    J = 0

    for s in range(nS):
        sum = 0
        for a in range(nA):
            sum = sum + r_s_a[s, a]*pi[s,a]
        J = J + d[s]*sum
    J = J/(1-gamma)
    return J

"""
    Utility function to get the expected discounted sum of returns computing d and r_s_a implicitly
    Args:    
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - pi (np.ndarray): the given policy [nS, nA]
        - reward (np.ndarray): the reward function [nS, nA, nS]
        - gamma (float): discount factor
        - mu (np.ndarray): initial state distribution [nS]

    return (float): the expected discounted sum of returns as a scalar value
"""
def get_expected_avg_reward(P_mat, pi, reward, gamma, mu):
    nS, nA = pi.shape
    r_s_a = compute_r_s_a(P_mat, reward)
    d = compute_d(mu, P_mat, pi, gamma)
    return compute_j(r_s_a, pi, d, gamma)

"""
    Compute the state action nextstate value function U(s, a, s') = R(s,a,s') + gamma * V(s')
    Args:
        - r_s_a (np.ndarray): the average reward when picking action a in state s [nS, nA]
        - gamma (float): discount factor
        - V (np.ndarray): the value function [nS]
    return (np.ndarray): the state action nextstate value function [nS, nA, nS]
"""
def compute_state_action_nextstate_value_function(r_s_a, gamma, V):
    nS, nA = r_s_a.shape
    U = np.zeros((nS, nA, nS))
    for s in range(nS):
        for a in range(nA):
            U[s, a] = r_s_a[s, a] + gamma*V
    return U

"""
    Utility function to get the state action nextstate value function U(s, a, s') = R(s,a,s') + gamma * V(s')
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - reward (np.ndarray): the reward function [nS, nA, nS]
        - gamma (float): discount factor
        - Q (np.ndarray): the state action value function [nS, nA]
        - det (bool): deterministic flag. Whether or not considering a deterministic policy
    return (np.ndarray): the state action nextstate value function [nS, nA, nS]
"""
def get_state_action_nextstate_value_function(P_mat, reward, gamma, Q, det=True):
    r_s_a = compute_r_s_a(P_mat, reward)
    V = get_value_function(Q, det)
    return compute_state_action_nextstate_value_function(r_s_a, gamma, V)

"""
    Extract the state action value function from a given state action nextstate value function Q = sum_s' P(s'|s,a) * U(s,a,s')
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - U (np.ndarray): state action next state value function as[nS, nA, nS]
    return (np.ndarray): the state action value function [nS, nA] 
"""
def rebuild_Q_from_U(P_mat, U):
    nS, nA, _ = U.shape
    Q_test = np.zeros((nS, nA))
    
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                Q_test[s, a] = Q_test[s,a] +  U[s, a, s_prime] * P_mat[s][a][s_prime]
    
    return Q_test

"""
    Compute the state policy advantage function A(s,a) = Q(s,a) - V(s)
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - V (np.ndarray): the value function [nS]
    return (np.ndarray): the policy advantage function [nS, nA]
"""
def compute_policy_advantage_function(Q, V):
    nS, nA = Q.shape
    A = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            A[s, a] = Q[s,a] - V[s]
    return A

"""
    Utility function to get the policy advantage function extracting V implicitly. A = Q - V
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the policy advantage function [nS, nA]
"""
def get_policy_advantage_function(Q, det=True):
    V = get_value_function(Q, det)
    return compute_policy_advantage_function(Q, V)

"""
    Compute the model advantage function A(s, a, s') = U(s, a, s') - Q(s, a)
    Args:
        - U (np.ndarray): the state action nextstate value function [nS, nA, nS]
        - Q (np.ndarray): the state action value function [nS, nA]
    return (np.ndarray): the model advantage function [nS, nA, nS]
"""
def compute_model_advantage_function(U, Q):
    nS, nA = Q.shape
    A = np.zeros((nS, nA, nS))

    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                A[s, a, s_prime] = U[s, a, s_prime] - Q[s, a]
    return A

"""
    Utility function to get the model advantage function computing U implicitly. A = U - Q
    Args:
        - P_mat (np.ndarray): probability transition function [nS, nA, nS]
        - reward (np.ndarray): the reward function [nS, nA, nS]
        - gamma (float): discount factor
        - Q (np.ndarray): the state action value function [nS, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the model advantage function [nS, nA, nS]
"""
def get_model_advantage_function(P_mat, reward, gamma, Q, det=True):
    U = get_state_action_nextstate_value_function(P_mat, reward, gamma, Q, det)
    return compute_model_advantage_function(U, Q)


"""
    Compute the relative policy advantage function A_pi_pi_prime(s)
    Args:
        - pi_prime (np.ndarray): the new policy to be compared [nS, nA]
        - A (np.ndarray): the policy advantage function [nS, nA]
    return (np.ndarray): the relative policy advantage function [nS]
"""
def compute_relative_policy_advantage_function(pi_prime, A):
    nS, nA = pi_prime.shape
    A_pi_prime = np.zeros(nS)
    for s in range(nS):
        for a in range(nA):
            A_pi_prime[s] = A_pi_prime[s] + pi_prime[s, a]*A[s, a]
    return A_pi_prime


"""
    Utility function to get the relative policy advantage function A_pi_pi_prime(s) = sum_a pi_prime(a|s) * A(s,a)
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - pi_prime (np.ndarray): the new policy to be compared [nS, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the relative policy advantage function [nS]
"""
def get_relative_policy_advantage_function(Q, pi_prime, det=True):
    A = get_policy_advantage_function(Q, det)
    return compute_relative_policy_advantage_function(pi_prime, A)

"""
    Utility function to get the relative policy advantage function A_pi_pi_prime(s) = sum_a (pi_prime(a|s) - pi(a|s)) * Q(s,a).
    Same result of get_relative_policy_advantage_function, but emphasizing the difference among the two policies.
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
        - pi_prime (np.ndarray): the new policy to be compared [nS, nA]
        - det (bool): deterministic flag. Whether or not extracting a deterministic policy
    return (np.ndarray): the relative policy advantage function [nS]
"""
def get_relative_policy_advantage_function_from_delta_p(Q, pi_prime, det=True):
    nS, nA = pi_prime.shape
    A_pi_prime = np.zeros(nS)
    pi = get_policy(Q, det)
    for s in range(nS):
        for a in range(nA):
            A_pi_prime[s] += (pi_prime[s, a] - pi[s,a]) * Q[s,a]
    return A_pi_prime



"""
    Compute the relative model advantage function A_tau_tauprime(s,a) = sum_s' P_tau_prime(s'|s,a) * A(s,a,s')
    Args:
        - P_mat_prime (nd.array): the probability transition function to be compared [nS, nA, nS]
        - A (np.ndarray): the model advantage function [nS, nA]
    return (np.ndarray): the relative model advantage function [nS, nA]
"""
def compute_relative_model_advantage_function(P_mat_prime, A):
    nS, nA, _ = A.shape
    rel_model_adv = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            rel_model_adv[s, a] = np.matmul(P_mat_prime[s][a], np.transpose(A[s, a]))
    return rel_model_adv

"""
    Compute the relative model advantage function hat. \hat{A}_tau(s,a) = sum_s' (P_tau(s'|s,a) - xi(s')) * U(s,a,s')
    N.B. to get the actual relative model advantage function you have to multiply times (\tau - \tau')
    Args:
        - P_mat (np.ndarray): probability transition function of the original problem, with tau=0 [nS, nA, nS]
        - xi (np.ndarray): state teleport probability distribution [nS]
        - U (np.ndarray): state action next state value function [nS, nA, nS]
    return (np.ndarray): the relative model advantage function hat [nS, nA]
"""
def compute_relative_model_advantage_function_hat(P_mat, xi, U):
    nS, nA, _ = U.shape
    model_adv_hat = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for s1 in range(nS):
                model_adv_hat[s, a] = model_adv_hat[s,a] + (P_mat [s][a][s1]- xi[s1])*U[s,a,s1]
    return model_adv_hat

"""
    Compute the relative model advantage function A_tau_tau_prime(s,a) = (tau - tau') * \hat{A}_tau(s,a).
    Args:
        - P_mat (np.ndarray): probability transition function of the original problem, with tau=0 [nS, nA, nS]
        - xi (np.ndarray): state teleport probability distribution [nS]
        - U (np.ndarray): state action next state value function [nS, nA, nS]
        - tau (float): discount factor
        - tau_prime (float): discount factor
    return (np.ndarray): the relative model advantage function [nS, nA]
"""
def compute_relative_model_advantage_function_from_delta_tau(P_mat, xi, U, tau, tau_prime):
    model_adv_hat = compute_relative_model_advantage_function_hat(P_mat, xi, U)
    return (tau - tau_prime)*model_adv_hat

"""
    Compute the discounted distribution relative model advantage function \mathcal{A}_tau_tauprime = sum_s sum_a delta(s,a) * A_tau_tauprime(s,a)
    Args:
        - A (np.ndarray): the model advantage function [nS, nA]
        - delta (np.ndarray): the discount state action distribution under policy pi [nS, nA]
    return (float): the discounted distribution relative model advantage function as a scalar
"""
def compute_discounted_distribution_relative_model_advantage_function(A, delta):
    expected_A = 0
    nS, nA = delta.shape
    for s in range(nS):
        for a in range(nA):
            expected_A = expected_A + A[s, a]* delta[s, a]
    return expected_A

"""
    Compute the discounted distribution relative model advantage function hat \mathcal{A}_tau_tauprime = sum_s sum_a delta(s,a) * \hat{A}_tau(s,a)
    N.B. to get the actual discounted distribution relative model advantage function you have to multiply times (\tau - \tau')
    Args:
        - A_tau_hat (np.ndarray): the relative model advantage function hat [nS, nA]
        - delta (np.ndarray): the discount state action distribution under policy pi [nS, nA]
    return (float): the discounted distribution relative model advantage function as a scalar
"""
def compute_discounted_distribution_relative_model_advantage_function_hat(A_tau_hat, delta):
    dis_rel_model_adv = 0
    nS, nA = A_tau_hat.shape
    for s in range(nS):
        for a in range(nA):
            dis_rel_model_adv = dis_rel_model_adv + A_tau_hat[s, a]*delta[s, a]
    return dis_rel_model_adv

"""
    Compute the relative model advantage function A_tau_tau_prime(s,a) = (tau - tau') * \hat{A}_tau(s,a).
    Args:
        - P_mat (np.ndarray): probability transition function of the original problem, with tau=0 [nS, nA, nS]
        - xi (np.ndarray): state teleport probability distribution [nS]
        - U (np.ndarray): state action next state value function [nS, nA, nS]
        - tau (float): discount factor
        - tau_prime (float): discount factor
    return (np.ndarray): the relative model advantage function [nS, nA]
"""
def compute_discounted_distribution_relative_model_advantage_function_from_delta_tau(A_tau_hat, delta, tau, tau_prime):
    dis_model_adv_hat = compute_discounted_distribution_relative_model_advantage_function_hat(A_tau_hat, delta)
    return (tau - tau_prime)*dis_model_adv_hat

# Here


"""
    Compute the expected difference between two probability transition functions P_tau and P_tau_prime
    Args:
        - P_mat_tau (np.ndarray): probability transition function [nS, nA, nS]
        - P_mat_tau_prime (np.ndarray): probability transition function [nS, nA, nS]
    return (float): the expected difference between the two probability transition functions
"""
def get_expected_difference_transition_models(P_mat_tau, P_mat_tau_prime):
    assert P_mat_tau.shape == P_mat_tau_prime.shape, "The two probability transition functions must have the same shape. Got P_mat_tau:{} and P_mat:{}".format(P_mat_tau.shape, P_mat_tau_prime.shape)
    de = 0
    nS_nA = P_mat_tau.shape[0]
    for i in range(nS_nA):
        de = de + np.linalg.norm(P_mat_tau_prime[i] - P_mat_tau[i], 1)
    return de/nS_nA

"""
    Compute the superior difference between two probability transition functions P_tau and P_tau_prime
    Args:
        - P_mat_tau (np.ndarray): probability transition function [nS, nA, nS]
        - P_mat_tau_prime (np.ndarray): probability transition function [nS, nA, nS]
    return (float): the superior difference between the two probability transition functions
"""
def get_sup_difference_transition_models(P_mat_tau, P_mat_tau_prime):
    assert P_mat_tau.shape == P_mat_tau_prime.shape, "The two probability transition functions must have the same shape. Got P_mat_tau:{} and P_mat:{}".format(P_mat_tau.shape, P_mat_tau_prime.shape)
    de = -np.inf
    nS_nA = P_mat_tau.shape[0]
    for i in range(nS_nA):
        norm = np.linalg.norm(P_mat_tau_prime[i] - P_mat_tau[i], 1)
        if norm > de:
            de = norm
    return de

"""
    Compute the superior difference between any two elements of the state action value function Q
    Args:
        - Q (np.ndarray): the state action value function [nS, nA]
    return (float): the superior difference between any two elements of the state action value function Q
"""
def get_sup_difference_q(Q):
    nS, nA = Q.shape
    sup = -np.inf
    for s in range(nS):
        for a in range(nA):
            for s1 in range(nS):
                for a1 in range(nA):
                    diff = abs(Q[s,a]-Q[s1,a1])
                    if diff > sup:
                        sup = diff
    return sup
    

"""
    Compute the performance improvement lower bound as l_b = A_hat*(tau-tau_1)/(1-gamma) - 2*gamma^2*Delta_Q*(tau-tau_1)^2/(2*(1-gamma)^2)
    Args:
        - A_hat (float): the discounted distribution relative model advantage function hat
        - gamma (float): discount factor
        - Delta_Q (float): the superior difference between any two elements of the state action value function Q
        - tau (float): discount factor
        - tau_1 (float): discount factor
    return (float): the performance improvement lower bound
"""
def compute_performance_improvement_lower_bound(A_hat, gamma, Delta_Q, tau:float, tau_1:float):
    return A_hat*(tau-tau_1)/(1-gamma) - 2*gamma**2*Delta_Q*(tau-tau_1)**2/(2*(1-gamma)**2)

"""
    Get the optimal tau prime as tau' = tau - A_hat*(1-gamma)/(4*gamma^2*Delta_Q)
    Args:
        - A_hat (float): the discounted distribution relative model advantage function hat
        - gamma (float): discount factor
        - tau (float): discount factor
        - Delta_Q (float): the superior difference between any two elements of the state action value function Q
    return (float): the optimal tau prime
"""
def compute_tau_prime(A_hat, gamma, tau, Delta_Q):
    return tau - A_hat*(1-gamma)/(4*gamma**2*Delta_Q)

"""
    Compute the optimal lower bound as A_hat^2/(8*gamma^2*Delta_Q)
    Args:
        - A_hat (float): the discounted distribution relative model advantage function hat
        - gamma (float): discount factor
        - Delta_Q (float): the superior difference between any two elements of the state action value function Q
    return (float): the optimal lower bound
"""
def compute_optimal_lower_bound(A_hat, gamma, Delta_Q):
    return A_hat**2/((8*gamma**2*Delta_Q))