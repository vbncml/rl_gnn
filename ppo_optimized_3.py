"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time
import gymnasium as gym
import itertools
import numpy as np
# import wandb
from network import FeedForwardNN, MLP, GCN, GAT, rGIN, GatedGraphNN, GCN_mean, GCN_new, GConvN, RecurrentGCN, RecurrentGCN_new, AttentionSTGCN, RecurrentTGCN, GCN_test
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, policy_class, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Make sure the environment is compatible with our code
        assert(type(env.observation_space) == gym.spaces.box.Box)
        assert(type(env.action_space) == gym.spaces.box.Box)
        
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        
        
        # self.obs_dim = env.observation_space.shape[1]
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        if policy_class in [GCN, GAT, GatedGraphNN, GCN_mean, GCN_new, GConvN, RecurrentGCN, RecurrentGCN_new, RecurrentTGCN, GCN_test]:
            self.edge_index = env.get_edge_index()
            self.obs_dim = env.observation_space.shape[1]

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        self.policy_network = policy_class

        # Initialize optimizers for actor and critic
        # if policy_class in [GCN, GAT, GatedGraphNN, GCN_mean, GCN_new]:
        #     self.actor_optim = Adam([
        #         {"params": self.actor.layer1.parameters(), "lr": 0},
        #         {"params": self.actor.layer2.parameters(), "lr": 0}
        #         ],
        #     lr=self.lr)
        #     self.critic_optim = Adam([
        #         {"params": self.critic.layer1.parameters(), "lr": 0},
        #         {"params": self.critic.layer2.parameters(), "lr": 0}
        #         ],
        #     lr=self.lr)
        #if policy_class==RecurrentGCN:
        #    self.actor_optim = Adam([
        #        {"params": self.actor.recurrent.parameters(), "lr": 0}
        #        ], 
        #    lr=self.lr)
        #    self.critic_optim = Adam([
        #        {"params": self.critic.recurrent.parameters(), "lr": 0}
        #        ], 
        #    lr=self.lr)
        # else:
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'lr': 0,
        }
    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()                     # ALG STEP 3
            
            # Calculate advantage using GAE
            
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 
            # print("A_k shape: ", A_k.shape)
            # print("Batch_obs:", batch_obs.shape)
            batch_size = batch_obs.shape[0]

            if self.policy_network in [GCN_test, GCN, GAT, GatedGraphNN, GCN_mean, GCN_new, GConvN, RecurrentGCN, RecurrentGCN_new, RecurrentTGCN]:
                data_list = []
                for i in range(batch_size):
                    data_list.append(Data(x=batch_obs[i], edge_index=self.edge_index))
                # V = torch.tensor(v_list, dtype=torch.float, requires_grad=True)
                # data_list = 
                loader = DataLoader(data_list, batch_size=batch_size)
                batch = next(iter(loader))
                # print("batch: ", batch)
                # # batch = Batch.from_data_list(data_list)
                # # print(batch)
                #V = self.critic(batch.x, batch.edge_index).squeeze()[-1]
                V = self.critic(batch.x, batch.edge_index).squeeze()
                #V = V.reshape(batch_size, -1)
                #V_mean = torch.mean(V,1)
                #V = V_mean
                #V = torch.mean(V)
                # V = self.critic(batch_obs, self.edge_index).squeeze()
            else:
                V = self.critic(batch_obs).squeeze()
            # V = self.critic(batch_obs).squeeze()
            # print("V shape: ", V.shape)
            batch_rtgs = A_k + V.detach()   
            
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of 
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger['lr'] = new_lr

                # Mini-batch Update
                np.random.shuffle(inds) # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation: 
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient descent easier behind the scenes.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy Regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss                    
                    
                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break # if kl aboves threshold
            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './centipede2_env_optim_ppo3_actor_GCN_test.pth')
                torch.save(self.critic.state_dict(), './centipede2_env_optim_ppo3_critic_GCN_test.pth')

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float)


    def rollout(self):
       
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            # Reset the environment. Note that obs is short for observation. 
            obs, _ = self.env.reset()
            # Initially, the game is not done
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render:
                    self.env.render()
                # Track done flag of the current state
                ep_dones.append(done)

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                if self.policy_network in [GCN_test, GCN, GAT, GatedGraphNN, GCN_mean, GCN_new, GConvN, RecurrentGCN, RecurrentGCN_new, RecurrentTGCN]:
                    val = self.critic(obs, self.edge_index)
                    #V_mean = torch.mean(V,1)
                    #V = V_mean
                    #val = torch.mean(val)
                else:
                    val = self.critic(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                # print(val.mean)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals,batch_dones

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action

        obs = torch.tensor(obs,dtype=torch.float)
        if self.policy_network in [GCN_test, GCN, GAT, rGIN, GatedGraphNN, GCN_mean,GCN_new,GConvN, RecurrentGCN, RecurrentGCN_new, RecurrentTGCN]:
            mean = self.actor(obs, self.edge_index)

        else:
            mean = self.actor(obs)
        # mean = self.actor(obs)
        # print("mean shape: ", mean)
        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()
        
        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        if self.deterministic:
            return mean.detach().numpy(), 1

        
        # Return the sampled action and the log probability of that action in our distribution
        
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        
        batch_size = batch_obs.shape[0]
        if self.policy_network in [GCN_test, GCN, GAT, GatedGraphNN, GCN_mean, GCN_new, GConvN, RecurrentGCN, RecurrentGCN_new, RecurrentTGCN]:
            data_list = []
            for i in range(batch_size):
                data_list.append(Data(x=batch_obs[i], edge_index=self.edge_index))
                # V = torch.tensor(v_list, dtype=torch.float, requires_grad=True)
                # data_list = 
            loader = DataLoader(data_list, batch_size=batch_size)
            batch = next(iter(loader))
                # print("batch: ", batch)
                # # batch = Batch.from_data_list(data_list)
                # # print(batch)
            V = self.critic(batch.x, batch.edge_index).squeeze()
            #print(V.shape)
            mean = self.actor(batch.x, batch.edge_index)
            mean = mean.reshape(batch_size, -1)
            #print(mean.shape)
            dist = MultivariateNormal(mean, self.cov_mat)

            #print(batch_acts.shape)

            log_probs = dist.log_prob(batch_acts)

        else:
            V = self.critic(batch_obs).squeeze()
            print(V.shape)
            mean = self.actor(batch_obs)
            print(mean.shape)
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_acts)
        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy()

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.0003                                 # Learning rate of actor optimizer
        self.gamma = 0.99                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = 0.95                                 # Lambda Parameter for GAE 
        self.num_minibatches = 6                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0                               # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.01                           # KL Divergence threshold
        self.max_grad_norm = 0.5                       # Gradient Clipping threshold
        self.writer = SummaryWriter(log_dir='logs/centipede2_env_GCN_test_{}'.format(time.time()))  # Save training log

        # Miscellaneous parameters
        self.render = False                             # If we should render during rollout
        self.save_freq = 10                             # How often we save in number of iterations
        self.deterministic = False                      # If we're testing, don't sample actions
        self.seed = 1234								# Sets the seed of our program, used for reproducibility of results

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

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        
        #Add results to summary writer
        self.writer.add_scalar('Timestep Reward', avg_ep_rews, t_so_far)
        self.writer.add_scalar('Timestep Loss', avg_actor_loss, t_so_far)
        self.writer.add_scalar('Timestep Episodic Length', avg_ep_lens, t_so_far)

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
