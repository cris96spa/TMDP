import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.envs.toy_text.utils import categorical_sample
from typing import List, Optional
import pygame
from RiverSwim import RiverSwim  # Ensure this is correctly imported based on your project structure
from stable_baselines3.common.callbacks import BaseCallback

class TMDP(Wrapper):
    def __init__(self, env: Env, xi: List[float], tau: float = 0.0, gamma: float = 0.99):
        """
        Initialize the TMDP wrapper.
        
        Parameters:
        env (Env): The environment to wrap.
        xi (List[float]): The probability distribution for teleportation states.
        tau (float): The probability of teleportation.
        gamma (float): The discount factor.
        """
        super(TMDP, self).__init__(env)
        self.xi = xi
        self.gamma = gamma
        self.update_tau(tau)
        self.reset()

    def step(self, action: int):
        """
        Take a step in the environment.
        
        Parameters:
        action (int): The action to take.
        
        Returns:
        tuple: A tuple containing the next state, reward, termination flag, truncation flag, and info dictionary.
        """
        self.env.lastaction = action

        if self.env.np_random.random() <= self.tau:
            # Teleport branch
            s_prime = categorical_sample(self.xi, self.env.np_random)
            r = 0
            self.env.s = s_prime
            if self.env.is_terminal(self.env.s):
                s_prime, _ = self.env.reset()
            truncated = False
            terminated = False
            info = {"teleport": True}
        else:
            s_prime, r, flags, info = self.env.step(action)
            terminated = flags.get("done", False)
            truncated = False
            r = r * (1 - self.tau)
            info["teleport"] = False

        if self.render_mode == "human":
            self.render()

        return s_prime, r, terminated, truncated, info

    def render(self):
        """Render the environment."""
        self.env.render()

    def update_tau(self, tau: float):
        """Update the teleportation probability."""
        self.tau = tau

    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
        Any: The initial state of the environment.
        """
        return self.env.reset(**kwargs)


class TeleportFilterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TeleportFilterCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        return True 
    

