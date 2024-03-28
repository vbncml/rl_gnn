import gymnasium as gym
import sys
import torch
import numpy as np
import custom_envs
from network import FeedForwardNN, MLP, GCN, GAT, rGIN
from ppo import PPO
from arguments import get_args
from eval_policy import eval_policy

def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training", flush=True)

    # model = PPO(policy_class=MLP, env=env, **hyperparameters)
    model = PPO(policy_class=GAT, env=env, **hyperparameters)

    if actor_model != '' and critic_model != '':
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training from scratch.", flush=True)

    model.learn(total_timesteps=10_000_000)

def test(env, actor_model):
    
    print(f"Testing {actor_model}", flush=True)

    if actor_model == '':
	    print(f"Didn't specify model file. Exiting.", flush=True)
	    sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    # # else:
    # obs_dim = env.observation_space.shape[0]
	
    act_dim = env.action_space.shape[0]

    policy = FeedForwardNN(obs_dim, act_dim)

    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy=policy, env=env, render=True)

def main(args):

    hyperparameters = {
                'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.1,
				'render': False,
				'render_every_i': 10
    }  

    if args.mode == 'train':
        env = gym.make('HumanoidG-v2')
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        env = gym.make('HumanoidG-v0', render_mode='rgb_array')
        test(env=env, actor_model=args.actor_model)

if __name__=='__main__':
    args = get_args()
    main(args)