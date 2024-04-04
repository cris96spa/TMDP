import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from DiscreteEnv import DiscreteEnv
from gymnasium.envs.toy_text.utils import categorical_sample
from river_swim_generator import generate_river
from typing import List, Optional

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
class River(DiscreteEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    """Constructor

        Args:
            nS (int): _description_
            small (int, optional): small reward. Defaults to 5.
            large (int, optional): large reward. Defaults to 10000.
            seed (float, optional): pseudo-random generator seed. Default to None.
    """
    def __init__(self, nS, mu, small=5, large=10000, seed=None, render_mode=None):
        
        self.nS = nS
        self.window_nS = 400
        self.window = None
        self.clock = None
        
        self.render_mode = render_mode

        P, P_mat, nA = self.generate_env(nS, small, large)
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


    def generate_env(self, nS, small, large):
        # Generate river parameters using the auxiliary function    
        nS, nA, p, r = generate_river(nS, small, large)

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
                    
                    done = self.is_terminal(s1)

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

        if self.render_mode == "human":
            self._render_frame()

        return int(self.s), {"prob":self.mu[int(self.s)]}
    
    """
        Check if the state is terminal
    """
    def is_terminal(self, state):
        return int(state) == self.nS-1
    

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

        if self.render_mode == "human":
            self._render_frame()

        return int(s), r, {"done":done}, {"prob": p}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_nS, 50)  # Make the window wider and less tall
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_nS, 50))  # Adjust canvas nS
        canvas.fill((255, 255, 255))
        pix_square_nS = self.window_nS / self.nS  # Adjust for single line

        # Draw the target at the last position
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_nS * (self.nS - 1),  # X coordinate at the far right
                0,  # Y coordinate, centered on the line
                pix_square_nS, pix_square_nS)  # Width and height of the rectangle
        )

        # Draw the agent as a circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int((self.s + 0.5) * pix_square_nS), 25),  # Centered on the line
            int(pix_square_nS / 3)
        )

        for x in range(self.nS + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_nS * x),
                (self.window_nS, pix_square_nS * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_nS * x, 0),
                (pix_square_nS * x, self.window_nS),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _rendeer_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_nS, self.window_nS)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_nS, 1))
        canvas.fill((255, 255, 255))
        pix_square_nS = self.window_nS / self.nS  # The nS of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_nS * (self.nS-1),
                (pix_square_nS, pix_square_nS),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.s + 0.5) * pix_square_nS,
            pix_square_nS / 3,
        )

        # Finally, add some gridlines
        for x in range(self.nS + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_nS * x),
                (self.window_nS, pix_square_nS * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_nS * x, 0),
                (pix_square_nS * x, self.window_nS),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()