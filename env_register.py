import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='TMDP-v1',
    entry_point='TMDP-v1:TMDP',
)
