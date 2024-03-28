import torch
import numpy as np
import gym
import time 

from network import FeedForwardNN, MLP, GCN, GAT, rGIN
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch import nn

class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        self._init_hypeparameters(hyperparameters)

        #Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[1]
        self.act_dim = env.action_space.shape[0]
        
        if policy_class in [GCN, GAT]:
            self.edge_index = env.get_edge_index()
        
        self.policy_network = policy_class
        # #Initialize actor and critic networks
        # self.actor = FeedForwardNN(self.obs_dim, self.act_dim) 
        # self.critic = FeedForwardNN(self.obs_dim, 1)

        self.actor = policy_class(self.obs_dim, self.act_dim) 
        self.critic = policy_class(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)

        # To store multivariate distirbution matrix where we can sample actions from
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        
        self.logger = {
            'delta_t': time.time_ns(),
            't': 0,          # timesteps so far
            'i': 0,          # iterations so far
            'batch_len': [],       # episodic lengths in batch
            'batch_rewards': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }
    
    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t = 0 # timesteps simulated
        i = 0 
        while t < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_len = self.rollout()

            # compute how many timesteps we collected for this batch
            t += np.sum(batch_len)
            i += 1

            #Logging
            self.logger['t'] = t
            self.logger['i'] = i
            # compute value of observation V
            
            V, _ = self.evaluate(batch_obs, batch_acts)

            # compute advantage A and normalize A
            A_k = batch_rewards_to_go - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iter):
                V, log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(log_probs - batch_log_probs)

                #surrogate loss 1
                surr_loss_1 = ratios*A_k
                #surrogate loss 2 clips ratios to make sure we are not stepping too far during gradient ascent
                #torch.clamp binds ratios between upper and lower bounds
                surr_loss_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)*A_k
                # actor loss
                actor_loss = (-torch.min(surr_loss_1, surr_loss_2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rewards_to_go)
                
                # calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())
            self._log_summary()

            if i % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor_dogv2_MLP.pth')
                torch.save(self.critic.state_dict(),'./ppo_critic_dogv2_MLP.pth')

    def rollout(self):
        #Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_len = []

        t = 0
        
        while t < self.timesteps_per_batch:
            
            episode_rewards = []
            
            obs, _ = self.env.reset()
            done = False

            for episode in range(self.timesteps_per_episode):
                if self.render and (self.logger['i'] % self.render_every_i == 0) and len(batch_len) == 0:
                    self.env.render()
                t += 1
                # collect obs
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                
                obs, reward, done, _, _ = self.env.step(action)
                
                # collect reward, action and log prob
                episode_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            #collect episodic length and rewards
            batch_len.append(episode+1)
            batch_rewards.append(episode_rewards)
        # print(batch_obs.shape)
        batch_obs = torch.tensor(np.array(batch_obs), dtype = torch.float)
        # print(batch_obs.shape)
        batch_acts = torch.tensor(batch_acts, dtype = torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)

        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        self.logger['batch_rewards'] = batch_rewards
        self.logger['batch_len'] = batch_len

        # return batch_obs[-1,:], batch_acts, batch_log_probs, batch_rewards_to_go, batch_len
        return batch_obs[-1,:], batch_acts, batch_log_probs, batch_rewards_to_go, batch_len

    def get_action(self, obs):
        
        # Same as caling self.actor.forward(obs)
        # print(self.policy_network)
        # print(self.actor)
        # print(GCN)
        if self.policy_network in [GCN, GAT, rGIN]:
            mean = self.actor(obs, self.edge_index)
        else:
            mean = self.actor(obs)
        # Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample action from distribution and get log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        #return action.detach().numpy().reshape(17,), log_prob.detach()
        return action.detach().numpy().reshape(8,), log_prob.detach()
        
    def compute_rewards_to_go(self, batch_rewards):
        #rewards per episode per batch shape: num_timesteps per episode
        batch_rewards_to_go = []

        for rewards_per_episode in reversed(batch_rewards):
            
            discounted_reward = 0
            for reward in reversed(rewards_per_episode):
                discounted_reward = reward + discounted_reward*self.gamma
                batch_rewards_to_go.insert(0, discounted_reward)
        
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)

        return batch_rewards_to_go
    
    def _init_hypeparameters(self, hyperparameters):
        self.timesteps_per_batch = 4800
        self.timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iter = 5
        self.clip = 0.1
        self.lr = 0.0003

        self.writer = SummaryWriter(log_dir='logs/dog-v2-MLP')
        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")
  
    def evaluate(self, batch_obs, batch_acts):
        # query critic network for a value V for each obs in batch_obs
        # squeeze changes dim of tensor for example [[1],[2],[3]] => [1,2,3]
        if self.policy_network in [GCN, GAT]:
            V = self.critic(batch_obs, self.edge_index).squeeze()
        else:
            V = self.critic(batch_obs).squeeze()
        # compute log probabilities of batch actions using most recent actor network
        if self.policy_network in [GCN, GAT, rGIN]:    
            mean = self.actor(batch_obs, self.edge_index)
        else:
            mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        # print(mean.shape)
        # print(batch_acts.shape)
        log_probs = dist.log_prob(batch_acts)

        # return predicted V and log probs
        return V, log_probs
    
    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values.
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t = self.logger['t']
        i = self.logger['i']
        avg_ep_lens = np.mean(self.logger['batch_len'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rewards']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        #Add results to summary writer
        self.writer.add_scalar('Timestep Reward', avg_ep_rews, t)
        self.writer.add_scalar('Timestep Loss', avg_actor_loss, t)
        self.writer.add_scalar('Timestep Episodic Length', avg_ep_lens, t)

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        
        # Reset batch-specific logging data
        self.logger['batch_len'] = []
        self.logger['batch_rewards'] = []
        self.logger['actor_losses'] = []


# import gym

# env = gym.make('Pendulum-v0')
# model = PPO(FeedForwardNN, env)
# model.learn(10000)

    

