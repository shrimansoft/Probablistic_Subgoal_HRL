import torch
from torch.distributions import multivariate_normal


#Implementing KL-Divergence using mean and variance from VAE
def KL_Divergence(Distribution_1,Distribution_2,device='cpu'):
    
    Mean_1 , std_1 = (Distribution_1["mean"]).to(device),(Distribution_1["std"]).to(device)
    Mean_2 , std_2 = (Distribution_2["mean"]).to(device),(Distribution_2["std"]).to(device)
    Mean_1 = Mean_1.unsqueeze(-1) #shape = (1,2,1)
    Mean_2 = Mean_2.unsqueeze(-1) #shape = (1,2,1)
    std_1 = std_1.unsqueeze(-1) #shape = (1,2,1)
    std_2 = std_2.unsqueeze(-1) #shape = (1,2,1)
    k=Mean_1.shape[-2]
    Covariance_1 = torch.eye(k).to(device)*(std_1**2) #shape = (1,2,2)
    Covariance_2 = torch.eye(k).to(device)*(std_2**2) #shape = (1,2,2)
    det_1 = torch.linalg.det(Covariance_1)
    det_2 = torch.linalg.det(Covariance_2)
    KL_Divergence = 0.5*(torch.log(torch.div(det_2,det_1)).unsqueeze(-1)
                         + torch.diagonal(torch.linalg.inv(Covariance_2)*Covariance_1,offset=0,dim1=-2,dim2=-1).sum(-1).unsqueeze(-1)
                         +torch.matmul(torch.transpose(Mean_2-Mean_1,dim0=-2,dim1=-1),torch.matmul(torch.linalg.inv(Covariance_2),(Mean_2-Mean_1))).squeeze(-2) -k)

    return KL_Divergence



