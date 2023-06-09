import numpy as np
from gym import Env
from scipy.special import softmax

"""
    Sample from categorical distribution
        @prob_n : probability distribution vector
        @np_random: random number generator
        return: a categorical state drown from prob_n
"""
def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    # Compute cumulative sum of the probability vector
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()

"""
    Compute the average reward when picking action a in state s
    @nS: number of states
    @nA: number of actions
    @P_mat: probability transition function
    @reward: the reward function
    return:the average reward when picking action a in state s as a |S| x |A| matrix
"""
def compute_r_s_a(nS, nA, P_mat, reward):
    # Average reward when taking action a in state s, of size |S|x|A|
    r_s_a =np.zeros(shape=(nS, nA))
    for s in range(nS):
        for a in range(nA):
            avg_rew = 0
            for s_prime in range(nS):
                avg_rew = avg_rew + P_mat[s*nA + a][s_prime] * reward[s, a, s_prime]
            #print("Updating state {}, action {}, with {}".format(s, a, avg_rew))
            r_s_a[s, a] = avg_rew
    return r_s_a

"""
    Compute the probability of moving from state s to state sprime, under policy pi
        @P_mat: probability transition function
        @pi: the given policy
        return: the probability of moving from state s to state sprime under policy pi, as a [nS x nS] matrix 
"""
def compute_p_sprime_s(P_mat, pi):
    nS, nA = pi.shape
    P_sprime_s = np.zeros((nS, nS))
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                P_sprime_s[s][s_prime] = P_sprime_s[s][s_prime] + pi[s, a] * P_mat[s*nA + a][s_prime]
    return P_sprime_s


"""
    Compute the discounted state distribution
        @mu: initial state distribution
        @P_mat: probability transition function
        @pi: the given policy
        @gamma: discount factor
        return: the discount state distribution as a vector of |S| elements
"""
def compute_d(mu, P_mat, pi, gamma):
    nS, nA = pi.shape
    I = np.eye(nS)
    P_sprime_s = compute_p_sprime_s(P_mat, pi)
    inv = np.linalg.inv(I - gamma*P_sprime_s)
    d = np.matmul((1-gamma)*mu, inv)
    return d

"""
    Compute the discounted state action distribution under policy pi
        @mu: initial state distribution
        @pi: the given policy
        return: the discount state action distribution under policy pi as a vector of |S|x|A| elements
"""
def compute_delta(d, pi):
    delta = np.zeros(pi.shape)
    nS, nA = pi.shape
    for s in range(nS):
        for a in range(nA):
            delta[s,a] = pi[s, a] * d[s]
    return delta

def get_delta(mu, P_mat, pi, gamma):
    d = compute_d(mu, P_mat, pi, gamma)
    return compute_delta(d, pi)

"""
    Extract the policy from a given state action value function
        @Q: the state action value function
        @det: deterministic flag. Whether or not extracting a deterministic policy
        return the greedy policy according to Q, as an |S|x|A| matrix
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
           
            # Provare normalizzando q sui valori di riga
    return pi


"""
    Extract the value function from a given state action value function
        @Q: the state action value function
        return the value function as V(s) = max{a in A}{Q(s,a)}
"""
def get_value_function(Q, det=True):
    pi = get_policy(Q, det)
    V = np.zeros(Q.shape[0])
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            V[s] = V[s] + Q[s,a]*pi[s,a]
    return V

def rebuild_Q_from_V(nS, nA, P_mat, reward, gamma, V):
    r_s_a = compute_r_s_a(nS, nA, P_mat, reward)
    Q_test = np.zeros((nS, nA))

    for s in range(nS):
        for a in range(nA):
            Q_test[s,a] = r_s_a[s,a] + gamma * np.matmul(P_mat[s*nA +a], np.transpose(V))
    return Q_test


"""
    Compute the expected discounted sum of returns
        @r_s_a: the average reward when picking action a in state s as an |S| x |A| matrix
        @pi: the given policy
        @gamma: discount factor
        return: the expected discounted sum of returns as a scalar value
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
    Utility function to get the expected discounted sum of returns
        @P_mat: probability transition function
        @pi: the given policy
        @reward: the reward function
        @gamma: discount factor
        return: the expected discounted sum of returns as a scalar value
"""
def get_expected_avg_reward(P_mat, pi, reward, gamma, mu):
    nS, nA = pi.shape
    r_s_a = compute_r_s_a(nS, nA, P_mat, reward)
    d = compute_d(mu, P_mat, pi, gamma)
    return compute_j(r_s_a, pi, d, gamma)

"""
    Compute the state action nextstate value function U(s, a, s')
        @nS: number of states
        @nA: number of actions
        @r_s_a: the average reward when picking action a in state s as an |S| x |A| matrix
        @gamma: discount factor
        @V: the state value function
        return: the state action nextstate value function as an |S|x|A|x|S| matrix
"""
def compute_state_action_nextstate_value_function(nS, nA, r_s_a, gamma, V):
    U = np.zeros((nS, nA, nS))
    for s in range(nS):
        for a in range(nA):
            U[s, a, :] = r_s_a[s, a] + gamma*V
    return U

"""
    Utility function to get the state action nextstate value function U(s, a, s')
        @P_mat: probability transition function
        @reward: the reward function
        @gamma: discount factor
        @Q: the state action value function
        return: the state action nextstate value function as an |S|x|A|x|S| matrix
"""
def get_state_action_nextstate_value_function(P_mat, reward, gamma, Q, det=True):
    nS, nA = Q.shape
    r_s_a = compute_r_s_a(nS, nA, P_mat, reward)
    V = get_value_function(Q, det)
    return compute_state_action_nextstate_value_function(nS, nA, r_s_a, gamma, V)

def rebuild_Q_from_U(P_mat, U):
    
    nS, nA, _ = U.shape
    Q_test = np.zeros((nS, nA))
    
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                Q_test[s, a] = Q_test[s,a] +  U[s, a, s_prime] * P_mat[s*nA + a][s_prime]
    
    return Q_test

"""
    Compute the state policy advantage function A(s,a)
        @nS: number of states
        @nA: number of actions
        @Q: state action value function
        @V: state value function
        return: the policy advantage function as an |S|x|A| matrix
"""
def compute_policy_advantage_function(nS, nA, Q, V):
    A = np.zeros(Q.shape)
    for s in range(nS):
        for a in range(nA):
            A[s, a] = Q[s,a] - V[s]
    return A

"""
    Utility function to get the policy advantage function A(s,a)
        @Q: the state action value function
        @det: deterministic flag. Whether or not extracting a deterministic policy
        return: the policy advantage function as an |S|x|A| matrix
"""
def get_policy_advantage_function(Q, det=True):
    nS, nA = Q.shape
    V = get_value_function(Q, det)
    return compute_policy_advantage_function(nS, nA, Q, V)

"""
    Compute the model advantage function A(s, a, s')
        @Q: the state action value function
        @U: state action next state value function as an |S|x|A|x|S| matrix
        return: the model advantage function as an |S|x|A|x|S| matrix
"""
def compute_model_advantage_function(U, Q):
    A = np.zeros(U.shape)
    nS, nA = Q.shape
    for s in range(nS):
        for a in range(nA):
            for s_prime in range(nS):
                A[s, a, s_prime] = U[s, a, s_prime] - Q[s, a]
    return A

"""
    Utility function to get the model advantage function A(s, a, s')
        @P_mat: probability transition function
        @reward: the reward function
        @gamma: discount factor
        @Q: the state action value function
        @det: deterministic flag. Whether or not extracting a deterministic policy
        return: the model advantage function as an |S|x|A|x|S| matrix
"""
def get_model_advantage_function(P_mat, reward, gamma, Q, det=True):
    U = get_state_action_nextstate_value_function(P_mat, reward, gamma, Q, det)
    return compute_model_advantage_function(U, Q)


"""
    Compute the relative policy advantage function A_pi_pi_prime(s)
        @pi_prime: the new policy to be compared
        @A: the policy advantage function as an |S|x|A| matrix
        return: the model advantage function as an |S| vector
"""
def compute_relative_policy_advantage_function(pi_prime, A):
    nS, nA = pi_prime.shape
    A_pi_prime = np.zeros(A.shape[0])
    for s in range(nS):
        for a in range(nA):
            A_pi_prime[s] = A_pi_prime[s] + pi_prime[s, a]*A[s, a]
    return A_pi_prime


"""
    Utility function to get the relative policy advantage function A_pi_pi_prime(s)
        @Q: the state action value function
        @pi_prime: the new policy to be compared
        @det: deterministic flag. Whether or not extracting a deterministic policy
        return: the relative policy advantage function as an |S| vector
"""
def get_relative_policy_advantage_function(Q, pi_prime, det=True):
    A = get_policy_advantage_function(Q, det)
    return compute_relative_policy_advantage_function(pi_prime, A)


"""
    Compute the relative model advantage function A_tau_tau_prime(s,a)
        @P_mat_prime: the probability transition function to be compared
        @A: the model advantage function as an |S|x|A|x|S| matrix
        return: the relative model advantage function as an |S|x|A| matrix
"""
def compute_relative_model_advantage_function(P_mat_prime, A):
    nS, nA, _ = A.shape
    A_tau_prime = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            A_tau_prime[s, a] = np.matmul(P_mat_prime[s*nA + a], np.transpose(A[s, a, :]))
    return A_tau_prime

"""
    Compute the relative model advantage function \hat{A}_tau_tau_prime(s,a)
    N.B. to get the actual relative model advantage function you have to multiply times (\tau - \tau')
        @P_mat_prime: the probability transition function to be compared
        @xi: state teleport probability distribution
        @U: state action next state value function as an |S|x|A|x|S| matrix
        return: the relative model advantage function hat as an |S|x|A| matrix
"""
def compute_relative_model_advantage_function_hat(P_mat, xi, U):
    nS, nA, _ = U.shape
    A_tau_prime = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for s1 in range(nS):
                A_tau_prime[s, a] = A_tau_prime[s,a] + (P_mat[s*nA +a][s1]-xi[s1])*U[s,a,s1]
            A_tau_prime[s, a] = A_tau_prime[s, a]
    return A_tau_prime

"""
    Compute the discounted distribution relative model advantage function A_tau_tau_prime
        @A: the relative model advantage function as an |S|x|A| matrix
        @delta: the discount state action distribution under policy pi as a vector of |S|x|A| elements
        return: the discounted distribution relative model advantage function as a scalar
"""
def compute_discounted_distribution_relative_model_advantage_function(A, delta):
    expected_A = 0
    nS, nA = delta.shape
    for s in range(nS):
        for a in range(nA):
            expected_A = expected_A + A[s, a]* delta[s, a]
    return expected_A

"""
    Compute the discounted distribution relative model advantage function hat \hat{A}^_tau_tau_prime
        N.B. to get the actual discounted distribution relative model advantage function you have to multiply times (\tau - \tau')
        @P_mat: probability transition function
        @xi: state teleport probability distribution
        @U: state action next state value function as an |S|x|A|x|S| matrix
        @delta: the discount state action distribution under policy pi as a vector of |S|x|A| elements
        return: the discounted distribution relative model advantage function hat as a scalar
"""
def compute_discounted_distribution_relative_model_advantage_function_hat(A_tau_hat, delta):
    expected_A = 0
    nS, nA = A_tau_hat.shape
    for s in range(nS):
        for a in range(nA):
            expected_A = expected_A + A_tau_hat[s, a]*delta[s, a]
    return expected_A


def get_expected_difference_transition_models(P_mat_tau, P_mat_tau_prime):
    assert P_mat_tau.shape == P_mat_tau_prime.shape
    de = 0
    nS_nA = P_mat_tau.shape[0]
    for i in range(nS_nA):
        de = de + np.linalg.norm(P_mat_tau_prime[i] - P_mat_tau[i], 1)
    return de/nS_nA

def get_sup_difference_transition_models(P_mat_tau, P_mat_tau_prime):
    assert P_mat_tau.shape == P_mat_tau_prime.shape
    de = -np.inf
    nS_nA = P_mat_tau.shape[0]
    for i in range(nS_nA):
        norm = np.linalg.norm(P_mat_tau_prime[i] - P_mat_tau[i], 1)
        if norm > de:
            de = norm
    return de


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
    

def compute_performance_improvement_lower_bound(A_hat, gamma, Delta_Q, tau:float, tau_1:float):
    return A_hat*(tau-tau_1)/(1-gamma) - 2*gamma**2*Delta_Q*(tau-tau_1)**2/(2*(1-gamma)**2)

def compute_tau_prime(A_hat, gamma, tau, Delta_Q):
    return tau - A_hat*(1-gamma)/(4*gamma**2*Delta_Q)

def compute_optimal_lower_bound(A_hat, gamma, Delta_Q):
    return A_hat**2/((8*gamma**2*Delta_Q))