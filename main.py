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
from Varitational_Autoencoder import VAE_representation_network
import os

os.add_dll_directory("C://Users//jaygu//.mujoco//mujoco210//bin")

env_args = Arguments.environment_args()
lower_agent_args = Arguments.Lower_level_args()
higer_agent_args = Arguments.Higher_level_args()
VAE_args = Arguments.VAE_args()
run_name = f"{int(time.time())}"           
writer = SummaryWriter(f"runs/{run_name}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_lower_reward(next_observation,subgoal,VAE_Network):
    next_observation = torch.tensor(next_observation['observation']).to(device)
    subgoal = torch.tensor(subgoal[-1])
    with torch.no_grad():
        Distribution_next = VAE_Network(next_observation)
    reward = -torch.norm(Distribution_next,subgoal)
    return reward.detach().cpu().numpy()
        
def is_subgoal_reached(next_observation,subgoal,threshold,VAE_Network):
    subgoal = torch.tensor(subgoal[1])
    achieved_goal = torch.tensor(next_observation).to(device)
    with torch.no_grad():
        Distribution_next = VAE_Network(achieved_goal)
        KL_Divergence = KL_Divergence(Distribution_next,subgoal)
    return KL_Divergence < threshold



def lower_rollout(env,lower_agent,observation, subgoal,VAE_network,evaluation=False):
    k=env_args.lower_horizon
    aggregate_reward = 0
    
    transitions = []

    for _ in range(k):
        action = lower_agent.get_action(torch.tensor(observation['observation']), torch.tensor(subgoal[0])).squeeze()
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        aggregate_reward += env_reward
        subgoal_achieved = is_subgoal_reached(next_obs['observation'],
                                               subgoal,env_args.KL_threshold,
                                               VAE_network)
        lower_reward = get_lower_reward(next_obs, subgoal,VAE_network)
        if evaluation is False:
            lower_agent.replay_buffer.add(obs['observation'],
                                        action,
                                        lower_reward,
                                        next_obs['observation'],
                                        done=subgoal_achieved,
                                        goal=subgoal)
            lower_agent.update()
            VAE_network.update(level='lower')
        transitions.append([obs,
                             action,
                               lower_reward,
                                 next_obs, subgoal,
                                   subgoal_achieved,
                                     info])
        obs = next_obs.copy()
        if(subgoal_achieved or terminated or truncated):
            break
    return obs, aggregate_reward, terminated, truncated, info, transitions

def higher_rollout(env,higher_agent,lower_agent,observation,goal,VAE_network,evaluation=False):
    k = env_args.higher_horizon

    for _ in range(k):
        subgoal = higher_agent.get_action()
        observation=observation
        obs,aggregate_reward,terminated,truncated,info,transitions = lower_rollout(env,
                                                                                   lower_agent,
                                                                                   observation,
                                                                                   subgoal,
                                                                                   VAE_network,
                                                                                   evaluation)
        done = terminated or truncated
        if evaluation == False:
            higher_agent.replay_buffer.add(observation['observation'],
                                           subgoal,
                                           aggregate_reward,
                                           obs['observation'],
                                           done=done ,goal=goal)
            higher_agent.update()
            VAE_network.update(level='higher')
        observation = obs


print(gym.envs.registry.keys())
env= gym.make(env_args.env_id,render_mode="human",max_episode_steps=1000)
observation, info = env.reset()
goal = observation['desired_goal']
higher_agent = Higher_Agent(env).init_agent()
lower_agent = Lower_Agent(env).init_agent()
VAE_Network = VAE_representation_network(env,VAE_args,lower_agent,higher_agent)

higher_rollout(env,higher_agent,lower_agent,observation,goal,VAE_Network,evaluation=False)



