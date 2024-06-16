from stable_baselines3.common.buffers import RolloutBuffer
import numpy as np
from gymnasium.spaces import Space
import torch as th

class TeleportRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size: int, observation_space: Space, 
                 action_space: Space, device: str = 'cpu',
                 gamma: float = 0.99, gae_lambda: float = 1.0, 
                 n_envs: int = 1):
        super().__init__(buffer_size, observation_space, action_space, 
                         device, gamma, gae_lambda, n_envs)
        self.teleport_flags = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def reset(self) -> None:
        super().reset()
        self.teleport_flags = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(self, *args, infos, **kwargs) -> None:
        # Handle teleport flags
        for idx, info in enumerate(infos):
            if info.get('teleport', False):
                self.teleport_flags[self.pos, idx] = True
        super().add(*args, **kwargs)

    
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
       
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            teleport = self.teleport_flags[step].astype(np.float32)

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            delta *= (1 - teleport) # Filtering out teleportation transitions
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam * (1-teleport)
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values


