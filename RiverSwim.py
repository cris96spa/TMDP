import numpy as np
import pygame
from gymnasium import Env, spaces, utils, logger
from gymnasium.utils import seeding
from DiscreteEnv import DiscreteEnv
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled

from RiverSwim_generator import generate_river
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import matplotlib.pyplot as plt

"""
    A river swim environment is an environment in which there are several sequential states and
    only two possible moves, left and right. It is assumed that left move is in the same direction of the river
    flow, hence it always lead to the immediatly left state, whereas the right move is done against the flow, 
    meaning that it has a ver'y small probability of leading to the left, a pretty high probability of remain'ing 
    in the same state and a small probability of moving to the right. The right you are able to move, the higher
    will be rewards.
    
    It presents the following attributes:
        - reward (np.ndarray): rewards associated to each action for each state [ns, nA, nS]
        - P_mat (np.ndarray): Matrix probability of moving from state s to s' (for each pairs (s,s') when picking action a (for each a) [nS, nA, nS]
        - allowed_actions (list): List of allowed action for each state

    Args:
        DiscreteEnv (gym.ENV): Implementation of a discrete environment, from the gym.ENV class.
"""
class RiverSwim(DiscreteEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
    
    """Constructor

        Args:
            nS (int): _description_
            small (int, optional): small reward. Defaults to 5.
            large (int, optional): large reward. Defaults to 10000.
            seed (float, optional): pseudo-random generator seed. Default to None.
    """
    def __init__(self, nS, mu, small=5, large=10000, seed=None, render_mode=None, r_shape=False):
        
        self.nS = nS

        self.nrow = 1
        self.ncol = nS
        self.lastaction = None
        #self.lastreward = None
        self.start_state = None
        P, P_mat, nA = self.generate_env(nS, small, large, r_shape)
        self.reward_range = (small, large)
        self.nA = nA 
        self.P_mat = P_mat
        self.P = P
        self.mu = mu
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(nS)

        self._actions_to_direction =  {
            0: np.array([-1]),  # left
            1: np.array([1]),  # right
        }

        self.seed(seed)
        self.reset(seed=seed)


        # pygame utils
        self.window_nS = (min(64 * self.ncol, 1024), min(64 * self.nrow, 512))
        self.cell_nS = (
            self.window_nS[0] // self.ncol,
            self.window_nS[1] // self.nrow,
        )
        self.window_surface = None
        self.window = None
        self.clock = None
        
        self.render_mode = render_mode
        self.big_goal_img = None
        self.small_goal_img = None
        self.start_img = None
        self.water_img = None
        self.player_images = None

    def generate_env(self, nS, small, large, r_shape):
        # Generate river parameters using the auxiliary function    
        nS, nA, p, r = generate_river(nS, small, large, r_shape)

        # Parameter initialization
        self.reward = r

        # Creating the dictionary of dictionary of lists that represents P
        P = {s: {a :[] for a in range(nA)} for s in range(nS)}
        # Probability matrix of the problem dynamics
        P_mat = np.zeros(shape=(nS, nA, nS))
        self.allowed_actions = []

        # Assigning values to P and P_mat
        for s in range(nS):
            # Add allowed actions (left, right) for each state
            self.allowed_actions.append([1,1])

            for a in range(nA):
                for s1 in range(nS):
                    # Get the probability of moving from s->s1, when action a is picked
                    prob = p[s][a][s1]
                    # Get the reward associated to the transition from s->s1, when a is picked
                    reward = r[s][a][s1]
                    
                    done = self.is_terminal(s1) and reward != 0

                    # Build P[s][a] that is a list of tuples, containint the probability of that move, the next state, the associated reward and a termination flag
                    # The termination flag is set to True if the reward is different from 0, meaning that the agent reached the goal
                    P[s][a].append((prob, s1, reward, done))

                    # Assign P_mat values
                    P_mat[s][a][s1] = prob
        return P, P_mat, nA

    def _get_obs(self):
        return {"state": self.s}

    def seed(self, seed=None):
        # set a random generator
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """
        Reset the environment to an initial state
    """
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # Get an initial state, from initial state distribution
        super().reset(seed=seed)
        self.s = categorical_sample(self.mu, self.np_random)
        self.lastaction = None
        #self.lastreward = None
        self.start_state = self.s
        return int(self.s), {"prob":self.mu[int(self.s)]}
    
    """
        Check if the state is terminal
    """
    def is_terminal(self, state):
        #if self.lastreward is None:
        return int(state) == self.nS-1 or int(state) == 0
        """else:
            return  (int(state) == self.nS-1 or int(state) == 0) and self.lastreward != 0""" 


    """
        Environment transition step implementation.
        Args:
            -a: the action to be executed
        return:
            next state, the immediate reward, done flag, the probability of that specific transition
    """
    def step(self, a):
        assert self.action_space.contains(a), "Action {} is not valid.".format(a)

        # Get the list of possible transitions from the current state, given the action a
        transitions = self.P[int(self.s)][a]
        # Get the probability of moving from s to every possible next state, while picking action a
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, done = transitions[i]

        # update the current state
        self.s = s
        # update last action
        self.lastaction = a

        return int(s), r, {"done":done}, {"prob": p}


    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return 
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)


    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("River Swim")
                self.window_surface = pygame.display.set_mode(self.window_nS)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_nS)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if self.water_img is None:
            file_name = path.join(path.dirname(__file__), "img/water_img.png")
            self.water_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.big_goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/big_price.png")
            self.big_goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.small_goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/small_price.png")
            self.small_goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.player_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.player_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_nS)
                for f_name in elfs
            ]
        
        cmap = plt.colormaps['coolwarm']
        """if self.reward_shape:
            reward_min = self.reward_range[0]
            reward_max = self.reward_range[1]"""

        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_nS[0], y * self.cell_nS[1])
                rect = (*pos, *self.cell_nS)

                self.window_surface.blit(self.water_img, pos)
                if x == 0: # small goal
                    self.window_surface.blit(self.small_goal_img, pos)
                elif x == self.nS-1: # big goal
                    self.window_surface.blit(self.big_goal_img, pos)
                elif x == self.start_state:
                    self.window_surface.blit(self.start_img, pos)
                
                """if self.reward_shape:# and desc[y][x] == b"H" or desc[y][x] == b"G":
                    reward = self.shaped_rewards[x, y] 
                    color = cmap((reward - reward_min) / (reward_max - reward_min))[:3]  # Normalize & RGB
                    reward_surface = pygame.Surface(self.cell_nS, pygame.SRCALPHA)
                    reward_surface.fill((int(255*color[0]), int(255*color[1]), int(255*color[2]), 128)) 
                    self.window_surface.blit(reward_surface, pos)"""

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_nS[0], bot_row * self.cell_nS[1])
        last_action = self.lastaction if self.lastaction is not None else 2
        elf_img = self.player_images[last_action]

        self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
