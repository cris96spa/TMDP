import math
import gymnasium as gym
from gymnasium import spaces
import torch 
import copy
import numpy as np
import torch.nn.functional as F
from model_functions import compute_tau_prime, compute_eps_model
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from TeleportRolloutBuffer import TeleportRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import time
import sys

class TeleportPMPO(PPO):
    def __init__(self, *args, max_batches=None, 
                 **kwargs):
        
        super(TeleportPMPO, self).__init__(*args, **kwargs)
        self.max_batches = max_batches
        self.total_timesteps_tracked = 0
        
    def learn(
        self: OnPolicyAlgorithm,
        total_timesteps: int,
        callback: BaseCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        check_convergence:bool=False,
        eps_threshold:float=1e-10,
        eps_shift:float=0.001,
        max_eps_model:float=1.,
        static_curriculum:bool=False,
    ) -> OnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        if static_curriculum and self.env.envs[0].tau > 0.:
            print("Static curriculum enabled. The agent will train until convergence to tau = 0.")
            # Compute the number of updates needed to reach tau = 0
            n_updates = total_timesteps // self.n_steps - 2
            # Compute the epsilon value that ensure convergence to tau = 0 in n_updates
            eps_model = compute_eps_model(self.gamma, self.env.envs[0].tau, n_updates)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            observations = self.rollout_buffer.observations

            # Convert observations to PyTorch tensors
            if not isinstance(observations, torch.Tensor):
                observations = torch.tensor(observations, dtype=torch.float32, device=self.policy.device)

            # Get old action probabilities
            with torch.no_grad():
                old_action_probs = self.policy.get_distribution(observations).distribution.probs

            # Train the agent
            self.train()
            
            # Get current action probabilities
            with torch.no_grad():
                new_action_probs = self.policy.get_distribution(observations).distribution.probs
            
            # Compute the D_inf distance between old and new policies
            d_inf_distance = self.calculate_d_inf_distance(old_action_probs, new_action_probs)
            
            if static_curriculum and self.env.envs[0].tau > 0.:
               self.static_update_teleport_rate(eps_model, log_interval, iteration)
            else:
                self.update_teleport_rate(d_inf_distance, eps_shift, max_eps_model,
                                       log_interval, iteration)
            
            # Stop training if convergence reached
            if check_convergence:
                if self.env.envs[0].tau == 0. and d_inf_distance < eps_threshold:
                    break
        
        callback.on_training_end()
        return self

    def update_teleport_rate(self, d_inf_distance, eps_shift, 
                             max_eps_model, log_interval, iteration):
        tau = self.env.envs[0].tau
        tau_prime = tau
        eps_model = 0.
        
        if tau > 0.:
            eps_model = min(eps_shift - d_inf_distance, max_eps_model)
            if eps_model > 0:
                gamma_eps_model = eps_model*self.gamma/(1-self.gamma)
                tau_prime = compute_tau_prime(self.gamma, tau, gamma_eps_model)
                
                # Update the teleportation probability
                if hasattr(self.env, 'envs'):
                    for env in self.env.envs:
                        env.update_tau(tau_prime)

        if log_interval is not None and iteration % log_interval == 0:
            self.logger.record("teleport/tau", tau)
            self.logger.record("teleport/tau_prime", tau_prime)
            if eps_model:
                self.logger.record("teleport/eps_model", eps_model)
            self.logger.record("teleport/d_inf_distance", d_inf_distance)

    def static_update_teleport_rate(self, eps_model, log_interval, iteration):
        tau = self.env.envs[0].tau
        tau_prime = tau
        
        if tau > 0:
            tau_prime = compute_tau_prime(self.gamma, tau, eps_model)
            
            # Update the teleportation probability
            if hasattr(self.env, 'envs'):
                for env in self.env.envs:
                    env.update_tau(tau_prime)
        
        # Log information
        if log_interval is not None and iteration % log_interval == 0:
            self.logger.record("teleport/tau", tau)
            self.logger.record("teleport/tau_prime", tau_prime)
            if eps_model:
                self.logger.record("teleport/eps_model", eps_model)
    
    def calculate_d_inf_distance(self, old_action_probs, new_action_probs):
        """
        Calculate the D_inf distance between old and new policies.
        """
        # Calculate the L1 norm difference for each state
        l1_norms = torch.norm(old_action_probs - new_action_probs, p=1, dim=-1)
        
        # Return the maximum L1 norm difference
        d_inf_distance = torch.max(l1_norms).item()
        
        return d_inf_distance

    def calculate_kl_distance(self, old_action_probs, new_action_probs):
        """
        Compare old and new policies using KL Divergence and Cosine Similarity
        """
        
        kl_divergence = F.kl_div(new_action_probs.log(), old_action_probs, reduction='batchmean').item()
    
        return kl_divergence

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log information about the training, like episode rewards and losses
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int(self.num_timesteps / time_elapsed)
        
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if self.ep_info_buffer:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", time_elapsed, exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        
        self.logger.dump(step=self.num_timesteps)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TeleportRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        assert isinstance(rollout_buffer, TeleportRolloutBuffer), "Rollout buffer should be an instance of TeleportRolloutBuffer"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, infos=infos)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
