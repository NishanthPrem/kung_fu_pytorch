#%% Importing the libraries

import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

#%% Creating the nueral network

class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3,3), stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2a = nn.Linear(128, action_size)
        self.fc2s = nn.Linear(128, 1)
    
    def forward(self, state):
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0]
        
        return action_values, state_value

#%% Setting up the environment

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, height=42, width=42, crop=lambda img:img, dim_order='pytorch', color=False, n_frames=4):
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.frame_stack = n_frames
        
        n_channels = 3 * n_frames if color else n_frames
        obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.frames = np.zeros(obs_shape, dtype=np.float32)
        
    def reset(self):
        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset()
        self.update_buffer(obs)
        return self.frames, info
    
    def observation(self, img):
        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        
        if not self.color:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32')/255
        if self.color:
            self.frames = np.roll(self.frames, shift=-3, axis=0)
        else:
            self.frames = np.roll(self.frames, shift=-1, axis=0)
        
        if self.color:
            self.frames[-3] = img
        else:
            self.frames[-1] = img
        return self.frames
    
    def update_buffer(self, obs):
        self.frames = self.observation(obs)
    
def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n

print(state_shape)
print(number_actions)
print(f'Action Names {env.env.env.get_action_meanings()}')

#%% Initializing the hyperparameters

learning_rate = 1e-4
gamma = 0.99
num_env = 10

#%% Implementing the A3C
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    