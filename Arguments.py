import argparse



def VAE_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',type=str,default='cuda:0',help="GPU device")
    parser.add_argument('--latent_dim',type=int,default=2,help="Dimension the latent vector")
    parser.add_argument('--hidden_dim_1',type=int,default=256,help='First Hidden Dimension')
    parser.add_argument('--hidden_dim_2',type=int,default=128,help='Second Hidden Dimension')
    parser.add_argument('--hidden_dim_3',type=int,default=64,help='Third Hidden Dimension')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate for varitaional autoencoder')
    args = parser.parse_args()
    return args

def environment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id',type=str,default='Ant-v4',help="Environment_name")
    parser.add_argument('--lower_horizon',type = int,default=10,help='Max steps for lower agent for a given subgoal')
    args = parser.parse_args()
    return args

def Lower_level_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_lr',type=float,default=0.0003,help="Learning rate fot the low level critic")
    parser.add_argument('--policy_lr',type=float,default=0.0003,help="Learning rate fot the low level actor")
    parser.add_argument('--autotune',type=bool,default=False,help="Whether to autotune entropy coefficient")
    parser.add_argument('--alpha',type=float,default=0.2,help="Entropy coeffcient")
    parser.add_argument('--gamma',type=float,default=0.99,help="Discount factor")
    parser.add_argument('--batch_size',type=int,default=256,help="Batch size")
    parser.add_argument('--target_network_frequency',type=int,default=1,help="")
    parser.add_argument('--tau',type=float,default=0.05,help="Polyak constant")
    args = parser.parse_args()

def Higher_level_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_lr',type=float,default=0.0003,help="Learning rate fot the high level critic")
    parser.add_argument('--policy_lr',type=float,default=0.0003,help="Learning rate fot the high level actor")
    parser.add_argument('--autotune',type=bool,default=False,help="Whether to autotune entropy coefficient")
    parser.add_argument('--alpha',type=float,default=0.2,help="Entropy coeffcient")
    parser.add_argument('--gamma',type=float,default=0.99,help="Discount factor")
    parser.add_argument('--batch_size',type=int,default=256,help="Batch size")
    parser.add_argument('--target_network_frequency',type=int,default=1,help="")
    parser.add_argument('--tau',type=float,default=0.05,help="Polyak constant")
    args = parser.parse_args()