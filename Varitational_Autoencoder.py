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

        return {'output':output,'mean':mean,'std':std}
    
class VAE_representation_network():
    def __init__(self,env):
        self.args = Arguments.VAE_args()
        self.env = env
        self.VAE_network = VAE(self.args.hidden_dim_1,
                               self.args.hidden_dim_2,
                               self.args.hidden_dim_3,
                               self.env.observation_space['observation'].shape,
                               self.args.latent_dim,self.args.device)
        self.optimizer = optim.Adam(list(self.VAE_network.parameters()), lr=self.args.lr)
        

    def get_distribution(self,state):
        return self.VAE_network(state)

    def update(self,state_1,state_2):
        state_1 = state_1.to(self.args.device)
        state_2 = state_2.to(self.args.device)
        representation_state_1 = self.VAE_network(state_1)
        representation_state_2 = self.VAE_network(state_2)
        Distribution_1 = {'mean':representation_state_1['mean'],'std':representation_state_1['std']}
        Distribution_2 = {'mean':representation_state_2['mean'],'std':representation_state_2['std']}
        Loss = KL_Divergence(Distribution_1,Distribution_2)
        self.optimizer.zero_grad()
        Loss.backward()
        self.optimizer.step()

        