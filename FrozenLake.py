import numpy as np
import pygame

from gymnasium import Env, spaces, utils, logger
from gymnasium.utils import seeding
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled

from DiscreteEnv import DiscreteEnv
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import matplotlib.pyplot as plt

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_nS: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_nS or c_new < 0 or c_new >= max_nS:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(nS: int = 8, p: float = 0.8, seed=None) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        nS: nS of each side of the grid
        p: probability that a tile is frozen

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_rand, seed = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_rand.choice(["F", "H"], (nS, nS), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, nS)
    return ["".join(x) for x in board]


class FrozenLakeEnv(DiscreteEnv):
    """
    Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H)
    by walking over the Frozen(F) lake.
    The agent may not always move in the intended direction due to the slippery nature of the frozen lake.


    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:

    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP

    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the nS of the map.
    For example, the 4x4 map has 16 possible observations.

    ### Rewards

    Reward schedule:
    - Reach goal(G): +1
    - Reach hole(H): 0
    - Reach frozen(F): 0

    ### Arguments

    ```
    gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    ```

    `desc`: Used to specify custom map for frozen lake. For example,

        desc=["SFFF", "FHFH", "FFFH", "HFFG"].

        A random generated map can be specified by calling the function `generate_random_map`. For example,

        ```
        from gym.envs.toy_text.frozen_lake import generate_random_map

        gym.make('FrozenLake-v1', desc=generate_random_map(nS=8))
        ```

    `map_name`: ID to use any of the preloaded maps.

        "4x4":[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ]

        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]

    `is_slippery`: True/False. If True will move in intended direction with
    probability of 1/3 else will move in either perpendicular direction with
    equal probability of 1/3 in both directions.

        For example, if action is left and is_slippery is True, then:
        - P(move left)=1/3
        - P(move up)=1/3
        - P(move down)=1/3

    ### Version History
    * v1: Bug fixes to rewards
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name:str="4x4",
        is_slippery:bool=True,
        seed = None,
        reward_shape:bool=False,
        num_bins:int=0,
        goal_reward:float=1.,
        shape_range=(-1, 0),
        dense_reward:bool=False
    ):
        if desc is None and map_name is None:
            desc = generate_random_map(seed=seed)
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.nA = nA = 4
        self.nS = nS = nrow * ncol
        self.lastaction = None
        self.lastreward = None
        self.mu = np.array(desc == b"S").astype("float64").ravel()
        self.mu /= self.mu.sum()
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.reward_shape = reward_shape
        self.goal_reward = goal_reward
        # Reward shaping function
        def reward_shaping(n, num_bins, shape_range=(-1, 0), goal_reward=1):
            rewards = np.zeros((n, n))
            goal = (n-1, n-1)
            max_distance = (goal[0] - 0) + (goal[1] - 0)
            
            # Calculate distances for each cell from the goal (Manhattan distance)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i, j] = abs(goal[0] - i) + abs(goal[1] - j)
                    #distances[i,j] = np.sqrt((goal[0] - i)**2 + (goal[1] - j)**2)
            
            # Determine bin edges
            bin_edges = np.linspace(0, max_distance, num_bins + 1)
            # Flip the bin, the lower the distance, the higher the reward
            bin_edges = np.flip(bin_edges)
            # Calculate rewards for each bin using linear interpolation
            bin_rewards = np.linspace(shape_range[0], shape_range[1], num_bins+1)
            
            # Assign rewards based on bins
            for i in range(n):
                for j in range(n):
                    distance = distances[i, j]
                    bin_index = np.digitize(distance, bin_edges, right=False)-1   # -1 because np.digitize starts from 1
                    bin_index = max(0, bin_index)  # Ensure bin_index is within range [0, num_bins-1]
                    rewards[i, j] = bin_rewards[bin_index]
            # Set reward for the goal cell
            rewards[goal[0], goal[1]] = goal_reward
            return rewards

        # Determine the number of bins for reward shaping
        self.num_bins = num_bins if num_bins > 0 else int(nrow/3)
        if self.reward_shape:
            self.reward_range = shape_range
            self.shaped_rewards = reward_shaping(nrow, self.num_bins, 
                                                 self.reward_range, goal_reward=goal_reward)
        else:
            self.reward_range = (0, 1)
            self.shaped_rewards = np.zeros((nrow, ncol))
        

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            reward = float(newletter == b"G")
            
            if self.reward_shape:
                if dense_reward:                                        # Reward for each step
                    reward = self.shaped_rewards[newrow, newcol] 
                    if bytes(newletter) in b"H":                        # Reward at the terminal state
                        reward = min(reward*nS + abs(0 - row) + abs(0 - col), self.shaped_rewards[-1,-2])
                else:                                                   
                    if terminated:                                      # Reward only at the terminal state
                        reward = self.shaped_rewards[newrow, newcol]

            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))
        
        def compute_models():
            P_mat = np.zeros((nS, nA, nS))
            reward = np.zeros((nS, nA, nS))
            allowed_actions = []
            for s in range(nS):
                allowed_actions.append([1,1,1,1])
                for a in range(nA):
                    for prob, next_state, r, _ in self.P[s][a]:
                        P_mat[s, a, next_state] = prob
                        reward[s, a, next_state] = r
            return P_mat, reward, allowed_actions

        self.P_mat, self.reward, self.allowed_actions = compute_models()

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.seed(seed)
        self.reset(seed=seed)

        self.render_mode = render_mode

        # pygame utils
        self.window_nS = (min(64 * ncol, 768), min(64 * nrow, 768))
        self.cell_nS = (
            self.window_nS[0] // self.ncol,
            self.window_nS[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def to_row_col(self, state):
            return state // self.ncol, state % self.ncol

    def is_terminal(self, state):
        row, col = self.to_row_col(state)
        return bytes(self.desc[row, col]) in b"GH"

    def step(self, a):
        transitions = self.P[int(self.s)][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, done = transitions[i]
        self.s = s
        self.lastaction = a
        self.lastreward = r

        if self.render_mode == "human":
            self.render()
        return (int(s), r, {"done": done}, {"prob": p})

    def seed(self, seed=None):
        # set a random generator
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.mu, self.np_random)
        self.lastaction = None
        self.lastreward = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return self._render_text()
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
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_nS)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_nS)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_nS
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_nS)
                for f_name in elfs
            ]
        
        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        
        cmap = plt.colormaps['coolwarm']
        if self.reward_shape:
            reward_min = self.reward_range[0]
            reward_max = self.reward_range[1]

        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_nS[0], y * self.cell_nS[1])
                rect = (*pos, *self.cell_nS)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)
                
                if self.reward_shape:# and desc[y][x] == b"H" or desc[y][x] == b"G":
                    reward = self.shaped_rewards[x, y] 
                    color = cmap((reward - reward_min) / (reward_max - reward_min))[:3]  # Normalize & RGB
                    reward_surface = pygame.Surface(self.cell_nS, pygame.SRCALPHA)
                    reward_surface.fill((int(255*color[0]), int(255*color[1]), int(255*color[2]), 128)) 
                    self.window_surface.blit(reward_surface, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_nS[0], bot_row * self.cell_nS[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
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

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()


    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

