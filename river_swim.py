import numpy as np
import pygame
from DiscreteEnv import DiscreteEnv
from river_swim_generator import generate_river

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
            gamma (float, optional): discount factor. Default to 1.
            small (int, optional): small reward. Defaults to 5.
            large (int, optional): large reward. Defaults to 10000.
            seed (float, optional): pseudo-random generator seed. Default to None.
    """

    def __init__(self, nS, mu, gamma=1., small=5, large=10000, seed=None, render_mode=None):

        self.size = nS
        self.window_size = 400
        
        self._actions_to_direction =  {
            0: np.array([-1]),  # left
            1: np.array([1]),  # right
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

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
    
                    # Build P[s][a] that is a list of tuples, containint the probability of that move, the next state, the associated reward and a termination flag
                    # The termination flag is set to True if the reward is different from 0, meaning that the agent reached the goal
                    P[s][a].append((prob, s1, reward, reward !=0))

                    # Assign P_mat values
                    P_mat[s][a][s1] = prob
                    
        self.P_mat = P_mat
        # Calling the superclass constructor to initialize other parameters
        super(River, self).__init__(nS, nA, P, mu, gamma, seed, render_mode=render_mode)

    def _render_frame(self):
        if self.window is None:
            return

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (x * pix_square_size, 0), (x * pix_square_size, self.window_size), width=2)
            pygame.draw.line(canvas, (0, 0, 0), (0, x * pix_square_size), (self.window_size, x * pix_square_size), width=2)

        # Draw the agent
        agent_position = self.s[0]
        pygame.draw.circle(canvas, (255, 0, 0), (int(agent_position * pix_square_size), self.window_size // 2), 10)

        # Draw the target on the leftmost state
        target_position_left = 0
        pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect(int(target_position_left * pix_square_size), self.window_size // 2 - 20, int(pix_square_size), 40))

        # Draw the target on the rightmost state
        target_position_right = self.size - 1
        pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect(int(target_position_right * pix_square_size), self.window_size // 2 - 20, int(pix_square_size), 40))

        self.window.blit(canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(30)  # Adjust the framerate as needed


    def render(self):
        self._render_frame()
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _target_location(self):
        return self.size - 1