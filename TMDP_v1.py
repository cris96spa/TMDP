import numpy as np
from model_functions import *
import pygame 
from RiverSwim import RiverSwim
from gymnasium.envs.toy_text.utils import categorical_sample
from typing import List, Optional
from gymnasium import Env, Wrapper, utils, logger
from stable_baselines3.common.callbacks import BaseCallback


class TMDP(Wrapper):
    def __init__(self, env:Env, xi, tau:float=0., gamma=0.99):
        super(TMDP, self).__init__(env)
        
        self.xi = xi
        self.gamma = gamma
        self.update_tau(tau)
        self.reset()

    def step(self, a):
        self.env.lastaction = a

        if self.env.np_random.random() <= self.tau:
            # Teleport branch
            s_prime = categorical_sample(self.xi, self.env.np_random)
            r = 0
            self.env.s = s_prime
            if self.env.is_terminal(self.env.s):
                s_prime, _ = self.env.reset()

            terminated = False
            truncated = True
            info = {"teleport": True}
        else:
            #print("Following regular probability transition function")
            s_prime, r, flags, _ = self.env.step(a)
            terminated = flags["done"]
            truncated = False
            r = r * (1 - self.tau)
            info = {}
        if self.render_mode == "human":
            self.render()

        return s_prime, r, terminated, truncated, info
    
    def render(self):
        self.env.render()

    def update_tau(self, tau):
        self.tau = tau

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class TeleportFilterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TeleportFilterCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        valid_indices = [i for i, info in enumerate(infos) if not info.get('teleport', False)]

        for key in ['mb_obs', 'mb_rewards', 'mb_dones', 'mb_actions', 'mb_values', 'mb_neglogpacs']:
            if key in self.locals:
                self.locals[key] = self.locals[key][valid_indices]

        return True