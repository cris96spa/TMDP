import pickle
import random
import numpy as np
import os

class ReplayBuffer:
    def __init__(self, max_size, input_shape=(1,), seed=None):
        self.seed = random.seed(seed)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.new_action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.traj_counter = 0
    
    def store_transition(self, state, action, reward, new_state, new_action, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.new_action_memory[index] = new_action
        self.done_memory[index] = done
        if done:
            self.traj_counter += 1

        self.mem_cntr += 1
    
    def end_trajectory(self):
        start_indx = (self.mem_cntr % self.mem_size) - 1
        if start_indx < 0:
            start_indx += self.mem_size
        self.done_memory[start_indx] = True
        self.traj_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        new_actions = self.new_action_memory[batch]
        done = self.done_memory[batch]
        return states, actions, rewards, new_states, new_actions, done
    
    def sample_last(self, batch_size=None):
        
        if batch_size is None:
            batch_size = min(self.mem_cntr, self.mem_size) # sample all elements in the buffer

        if self.mem_cntr < batch_size:
            raise ValueError("Not enough elements in the buffer to sample a full batch")

        start_indx = (self.mem_cntr % self.mem_size) - batch_size
        if start_indx < 0:
            start_indx += self.mem_size
        
        batch = [(start_indx + i) % self.mem_size  for i in range(batch_size)]
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        new_actions = self.new_action_memory[batch]
        done = self.done_memory[batch]
        return states, actions, rewards, new_states, new_actions, done

    def clear(self):
        self.mem_cntr = 0
        input_shape = self.input_shape
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.new_action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.traj_counter = 0

    def len(self):
        return min(self.mem_cntr, self.mem_size)

    def save_buffer(self, env_name, suffix="", path="./checkpoints"):
        if not os.path.exists(path):
            os.makedirs(path)
        
        if path == "./checkpoints":
            path = "./checkpoints/replay_buffer_{}_{}".format(env_name, suffix)
        print("Saving replay buffer to {}".format(path))

        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load_buffer(self, path):
        print("Loading replay buffer from {}".format(path))
        with open(path, 'rb') as f:
            replay_buffer = pickle.load(f)
