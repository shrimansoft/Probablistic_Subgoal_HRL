import  Arguments 
import torch
import torch.nn as nn
import torch.optim as optim
from Loss import KL_Divergence

class VAE(nn.Module):
    def __init__(self,hidden_dim_1,hidden_dim_2,hidden_dim_3,input_dim,latent_dim,out_dim,device):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=hidden_dim_1),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_1,out_features=hidden_dim_2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_2,out_features=hidden_dim_3),
            nn.LeakyReLU()

        )

        self.mean = nn.Linear(in_features=hidden_dim_3,out_features=latent_dim)
        self.log_std = nn.Linear(in_features=hidden_dim_3,out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim,out_features=hidden_dim_1),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_1,out_features=hidden_dim_2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim_2,out_features=out_dim),
            nn.Tanh()
        )
        self.device = device
        self.LOG_STD_MAX =  1
        self.LOG_STD_MIN = -1

    def reparameterization(self,mean,std):
        epsilon = torch.randn_like(mean).to(self.device)
        z= mean + std*epsilon
        return z
    
    def forward(self,state):
        hidden_representation = self.encoder(state)
        mean = self.mean(hidden_representation)
        log_std  =self.log_std(hidden_representation)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        z = self.reparameterization(mean=mean,std=std)
        output = self.decoder(z)

        return output,{'mean':mean,'std':std}
    
class VAE_representation_network():
    def __init__(self,env,args,lower_agent,higher_agent):
        self.args = args
        self.lower_agent = lower_agent
        self.higher_agent = higher_agent
        self.env = env
        self.VAE_network = VAE(self.args.hidden_dim_1,
                               self.args.hidden_dim_2,
                               self.args.hidden_dim_3,
                               self.env.observation_space['observation'].shape,
                               self.args.latent_dim,self.args.device)
        self.optimizer = optim.Adam(list(self.VAE_network.parameters()), lr=self.args.lr)
        

    def get_distribution(self,state):
        return self.VAE_network(state)

    def update(self,level):
        if level=='higher':
            observations, actions, rewards, next_observations, goals, dones = self.higher_agent.replay_buffer.sample(self.args.batch_size)
            state_1 = observations
            state_2 = next_observations
        else:
            observations, _, _, next_observations,_,_ = self.lower_agent.replay_buffer.sample(self.args.batch_size)
            state_1 = observations
            state_2 = next_observations  
        representation_1 = self.VAE_network(state_1)
        representation_2 = self.VAE_network(state_2)
        representation_state_1 =representation_1[-1]
        representation_state_2 = representation_2[-1]
        output_1 = representation_1[0]
        output_2 = representation_2[0]
        reconstruction_loss = torch.norm(state_1-output_1,dim=1)+torch.norm(state_2-output_2,dim=1)
        Distribution_1 = {'mean':representation_state_1['mean'],'std':representation_state_1['std']}
        Distribution_2 = {'mean':representation_state_2['mean'],'std':representation_state_2['std']}
        if level=='higher':
            Loss = torch.max(0,self.args.m-KL_Divergence(Distribution_1,Distribution_2))+reconstruction_loss
        else :
            Loss = KL_Divergence(Distribution_1,Distribution_2)+reconstruction_loss
        self.optimizer.zero_grad()
        Loss.backward()
        self.optimizer.step()

        