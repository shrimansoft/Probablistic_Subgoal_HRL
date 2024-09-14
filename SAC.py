import torch
import numpy as np
import gymnasium as gym
import Varitational_Autoencoder as VAE

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from Replay_Buffer import ReplayBuffer

class SAC_Agent():

    def __init__(self, Actor_args, QNetwork_args, device, q_lr, policy_lr, autotune, alpha, gamma, target_network_frequency, tau,
                 action_space, batch_size):
        self.gamma = gamma
        self.actor = Actor(**Actor_args).to(device)
        self.qf1 = SoftQNetwork(**QNetwork_args).to(device)
        self.qf2 = SoftQNetwork(**QNetwork_args).to(device)
        self.qf1_target = SoftQNetwork(**QNetwork_args).to(device)
        self.qf2_target = SoftQNetwork(**QNetwork_args).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)
        self.autotune = autotune
        self.tau = tau
        self.action_space = action_space
        self.batch_size = batch_size
        if autotune:
            self.target_entropy = -Actor_args['action_dim']
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

        self.target_network_frequency = target_network_frequency
        self.replay_buffer = ReplayBuffer(observation_space_dim=[Actor_args['obs_dim']],
                                            action_space_dim=[Actor_args['action_dim']],
                                            goal_space_dim=[Actor_args['goal_dim']],
                                            buffer_size=100000)

    def get_action(self, obs, goal, eps, deterministic=False):
        if (deterministic == True):
            return self.actor(obs, goal)[0].detach().cpu().numpy()
        return self.actor.get_action(obs, goal)[0].detach().cpu().numpy()

    def update(self):
        observations, actions, rewards, next_observations, goals, dones = self.replay_buffer.sample(self.batch_size)
        next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_observations, goals)
        qf1_next_target = self.qf1_target(next_observations, next_state_actions, goals)
        qf2_next_target = self.qf2_target(next_observations, next_state_actions, goals)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(observations, actions, goals).view(-1)
        qf2_a_values = self.qf2(observations, actions, goals).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        qf_loss = qf_loss.double()
        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()


        pi, log_pi, _ = self.actor.get_action(observations, goals)
        qf1_pi = self.qf1(observations, pi, goals)
        qf2_pi = self.qf2(observations, pi, goals)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.get_action(observations, goals)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            # update the target networks
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, action_scale, action_bias,level):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.level =  level
        self.action_scale, self.action_bias = action_scale.to(self.device), action_bias.to(self.device)
        self.LOG_STD_MAX =  1
        self.LOG_STD_MIN = -1
        self.double()

    def forward(self, x, g):
        x, g = x.to(self.device), g.to(self.device)
        if(x.ndim == 1):
            x, g = x.unsqueeze(0), g.unsqueeze(0)
        x = F.relu(self.fc1(torch.cat([x, g], dim=-1)))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, g):
        mean, log_std = self(x, g)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        if(self.level=='lower'):
            action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + goal_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.double()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, a, g):
        x, a, g = x.to(self.device), a.to(self.device), g.to(self.device)
        x = torch.cat([x, a, g], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x