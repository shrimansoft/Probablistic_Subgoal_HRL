import torch
import numpy as np
import tensorboard
import gymnasium as gym
import time
import Arguments
from Lower_Level_Agent import Lower_Agent
from Higher_Level_Agent import Higher_Agent
from Loss import KL_Divergence
from torch.utils.tensorboard import SummaryWriter

environment_args = Arguments.environment_args()
lower_agent_args = Arguments.Lower_level_args()
higer_agent_args = Arguments.Higher_level_args()
VAE_args = Arguments.VAE_args()

env= gym.make(environment_args.env_id,render_mode="human")
env.reset()
device = 'cuda:0'

def get_lower_reward(next_observation,subgoal,VAE_Network):
    next_observation = torch.tensor(next_observation['observation']).to(device)
    subgoal = torch.tensor(subgoal[-1])
    with torch.no_grad():
        Distribution_next = VAE_Network(next_observation)
    reward = -torch.norm(Distribution_next,subgoal)
    return reward.detach().cpu().numpy()
        
def is_subgoal_reached(observation,goal,threshold):
    pass
def is_goal_reached():
    pass

def lower_rollout(env,lower_agent,observation, subgoal,env_args,VAE_network,evaluation=False):
    k=env_args.lower_horizon
    aggregate_reward = 0
    
    transitions = []

    for _ in range(k):
        action = lower_agent.get_action(torch.tensor(observation['observation']), torch.tensor(subgoal)).squeeze()
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        aggregate_reward += env_reward
        subgoal_achieved = is_subgoal_reached(next_obs['achieved_goal'], subgoal,environment_args.KL_threshold)
        lower_reward = get_lower_reward(next_obs, subgoal,VAE_network)
        if evaluation is False:
            lower_agent.replay_buffer.add(obs['observation'], action, lower_reward, next_obs['observation'], done=subgoal_achieved, goal=subgoal)
            lower_agent.update()
            VAE_network.update(level='lower')
        transitions.append([obs, action, lower_reward, next_obs, subgoal, subgoal_achieved, info])
        obs = next_obs.copy()
        if(subgoal_achieved or terminated or truncated):
            break
    return obs, aggregate_reward, terminated, truncated, info, transitions

def higher_rollout(env,higher_agent,lower_agent,observation,goal,env_args,VAE_network,evaluation=False):
    k = env_args.higher_horizon

    for _ in range(k):
        subgoal = higher_agent.get_action()
        observation=observation
        obs,aggregate_reward,terminated,truncated,info,transitions = lower_rollout(env,lower_agent,observation,subgoal,env_args,VAE_network,evaluation)
        done = is_goal_reached()
        if evaluation == False:
            higher_agent.replay_buffer.add(observation['observation'],subgoal,aggregate_reward,obs['observation'],done=done ,goal=goal)
            higher_agent.update()
            VAE_network.update(level='higher')
        observation = obs








