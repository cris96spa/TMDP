import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='TMDP-v0',
    entry_point='TMDP:TMDP',
)

def make_env(env_id, rank, env, xi, tau:float=0., 
             gamma:float=.99, seed=None, render_mode=None):
    def _init():
        e = gym.make(env_id, env=env, xi=xi, tau=tau, gamma=gamma, 
                     seed=seed, render_mode=render_mode)
        e.seed(seed + rank)
        return e
    return _init