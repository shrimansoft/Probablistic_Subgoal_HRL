import torch
import numpy as np
import tensorboard
import gymnasium as gym
import time
import Arguments
from Lower_Level_Agent import Lower_Agent
from Higher_Level_Agent import Higher_Agent
from Loss import KL_Divergence

environment_args = Arguments.environment_args()
env= gym.make(environment_args.env_id,render_mode="human")
env.reset()
env.render()
time.sleep(5)







