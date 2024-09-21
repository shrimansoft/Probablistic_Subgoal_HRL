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
from torch.utils.tensorboard import SummaryWriter

os.add_dll_directory("C://Users//jaygu//.mujoco//mujoco210//bin")
env_args = Arguments.environment_args()
lower_agent_args = Arguments.Lower_level_args()
higer_agent_args = Arguments.Higher_level_args()
VAE_args = Arguments.VAE_args()
run_name = f"{int(time.time())}"           
writer = SummaryWriter(f"runs/{run_name}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
total_timesteps = 0

def get_lower_reward(next_observation,subgoal,VAE_Network):
    next_observation = torch.tensor(next_observation).unsqueeze(0).to(device) # shape = (1,27)
    with torch.no_grad():
        Distribution_next = VAE_Network.get_distribution(next_observation)
    reward = -KL_Divergence(Distribution_next[-1],subgoal)
    return reward.detach().cpu().numpy()
        
def is_subgoal_reached(next_observation,subgoal,threshold,VAE_Network):
    achieved_goal = torch.tensor(next_observation).unsqueeze(0).to(device) #shape = (1,27)
    with torch.no_grad():
        Distribution_next = VAE_Network.get_distribution(achieved_goal)
        #subgoal['mean'].shape =(1,2) 
        #subgoal['std'].shape =(1,2)
        #Distribution_next[-1]['mean'].shape = (1,2)
        #Distribution_next[-1]['std'].shape = (1,2)
        KL_divergence = KL_Divergence(Distribution_next[-1],subgoal) # shape = (1,1)
    return (KL_divergence < threshold).detach().cpu().numpy()



def lower_rollout(env,lower_agent,observation, subgoal,VAE_network,writer,evaluation=False):
    k=env_args.lower_horizon
    aggregate_reward = 0
    subgoal_sample =  torch.tensor(subgoal[0]).reshape(-1,1).squeeze(-1) #shape = (2)
    obs = observation
    observation = torch.tensor(observation) #shape = (27)

    transitions = []

    for i in range(k):
        action = lower_agent.get_action(observation,subgoal_sample)[0].squeeze(0) # shape = (8)
        action = action.detach().cpu().numpy()
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        aggregate_reward += env_reward
        subgoal_achieved = is_subgoal_reached(next_obs['observation'],subgoal[-1],env_args.KL_threshold,VAE_network)
        lower_reward = get_lower_reward(next_obs['observation'], subgoal[-1],VAE_network)
        if evaluation is False:
            lower_agent.replay_buffer.add(obs,
                                        action,
                                        lower_reward,
                                        next_obs['observation'],
                                        done=subgoal_achieved,
                                        goal=subgoal[0].detach().cpu().numpy())
            lower_agent.update()
            VAE_network.update(level='lower')
        global total_timesteps
        total_timesteps = total_timesteps +1
        writer.add_scalar("data/lower_reward",lower_reward.item(), total_timesteps)
        transitions.append([obs,action,lower_reward,next_obs, subgoal,subgoal_achieved,info])
        obs = next_obs['observation']
        if subgoal_achieved:
            break
        if terminated or truncated:
            break
    return obs, aggregate_reward, terminated, truncated, info, transitions

def higher_rollout(env,higher_agent,lower_agent,observation,goal,VAE_network,writer,evaluation=False):
    k = env_args.higher_horizon
    obs = observation['observation']
    observation = torch.tensor(observation['observation'])
    goal = torch.tensor(goal)
    for _ in range(k):
        subgoal = higher_agent.get_action(observation,goal)
        # goal shape = (2)
        # observation['observation'] shape = (27)
        # subgoal[0] shape = (1,2)
        # subgoal[1]['mean'] shape = (1,2)
        # subgoal[1]['std] shape = (1,2)

        next_obs,aggregate_reward,terminated,truncated,info,transitions = lower_rollout(env,lower_agent,observation,subgoal,VAE_network,writer,evaluation)
        writer.add_scalar("data/Higher_reward",aggregate_reward, total_timesteps//env_args.lower_horizon)
        
        done = terminated or truncated 
        if evaluation == False:
            higher_agent.replay_buffer.add(obs,
                                           subgoal[0].detach().cpu().numpy(),
                                           aggregate_reward,
                                           next_obs,
                                           done=done ,goal=goal.detach().cpu().numpy())
            higher_agent.update()
            VAE_network.update(level='higher')
        obs = next_obs
        if done:
            break


OPEN = [[1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 'g', 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 'r', 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]]
env= gym.make(env_args.env_id,render_mode="human",max_episode_steps=500,maze_map = OPEN)


higher_agent_class = Higher_Agent(env)
higher_agent = higher_agent_class.init_agent()
lower_agent_class = Lower_Agent(env)
lower_agent = lower_agent_class.init_agent()
VAE_Network = VAE_representation_network(env,VAE_args,lower_agent,higher_agent,device)
for i in range(env_args.episodes):
    observation, info = env.reset()
    env.render()
    goal = observation['desired_goal']
    higher_rollout(env,higher_agent,lower_agent,observation,goal,VAE_Network,writer,evaluation=False)



