"""
# TMDP: Teleport Markov Decision Process
![alt text](teleport_demo.gif)
# Curriculum Learning through Teleportation: The Teleport MDP

## Introduction

Deep Reinforcement Learning (DRL) has revolutionized complex decision-making tasks, but still faces challenges in environments with sparse rewards, high-dimensional spaces, and long-term credit assignment issues. This project introduces the Teleport Markov Decision Processes (TMDPs) framework, which enhances the exploration capabilities of RL agents through a teleportation mechanism, contributing to more effective curriculum learning.

## The Teleport MDP Framework

### What is a Teleport MDP?

A Teleport MDP extends the traditional Markov Decision Process (MDP) by adding a teleportation mechanism. It allows an agent to be relocated to any state during an episode, controlled by:

- Teleport rate (τ): Determines the frequency of teleportation
- State teleport probability distribution (ξ): Dictates the possible states for teleportation

### How It Works

TMDPs start with a high teleport rate for wide exploration, gradually reducing it to increase task complexity and converge towards the original problem formulation.

### Mathematical Formulation

A TMDP is defined by the tuple M=⟨S,A,P,R,γ,μ,τ,ξ⟩, where:

- S: State space
- A: Action space
- P(s′∣s,a): Transition probability model
- R(s,a): Reward function
- γ: Discount factor
- μ: Initial state distribution
- τ: Teleport rate
- ξ: Teleport probability distribution

The transition model in TMDP is defined as:

Pτ(s′∣s,a)=(1−τ)P(s′∣s,a)+τξ(s′)

## Practical Algorithms

We developed several algorithms integrating teleport-based curricula:

1. Teleport Model Policy Iteration (TMPI)
2. Static Teleport (S-T)
3. Dynamic Teleport (D-T)

## Experimental Evaluation

We conducted experiments using two RL environments:

1. Frozen Lake
2. River Swim

Results demonstrated that TMDP-based algorithms consistently outperformed their vanilla counterparts in both environments.

## Conclusion

The Teleport MDP framework offers a flexible and effective approach to curriculum design in reinforcement learning, reducing reliance on domain-specific expertise and improving learning efficiency.

## Co-Authors

This research was conducted in collaboration with:

- Prof. Marcello Restelli
- Dr. Alberto Maria Metelli
- Dr. Luca Sabbioni

## References

1. Andrychowicz, M., et al. (2017). Hindsight experience replay.
2. Florensa, C., et al. (2017). Reverse curriculum generation for reinforcement learning.
3. Kakade, S. M., & Langford, J. (2002). Approximately optimal approximate reinforcement learning.
4. Metelli, A. M., et al. (2018). Configurable Markov decision processes.
5. Schulman, J., et al. (2017). Proximal policy optimization algorithms.
6. Bengio, Y., et al. (2009). Curriculum learning.