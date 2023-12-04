import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(N_frame=4, test=False)
		self.test_env = CarRacingEnvironment(N_frame=4, test=True)
		
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)# modify 
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)# modify 
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())# modify 
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)# modify 

		# choose Gaussian noise or OU noise

		# noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		# noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		# self.noise = OUNoiseGenerator(noise_mean, noise_std)

		self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		
	
	def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		with torch.no_grad():
			state = torch.FloatTensor(state).unsqueeze(0)
			#print(state.to(self.device).shape)
			action = self.actor_net(state.to(self.device), brake_rate)
			action = action.cpu().numpy().squeeze()
			action += (self.noise.generate() * sigma) # modify

		return action

		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		### TD3 ###
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		## Update Critic ##
		# critic loss
		q_value1 = self.critic_net1(state, action)
		q_value2 = self.critic_net2(state, action)# modify 
		with torch.no_grad():
		# 	# select action a_next from target actor network and add noise for smoothing
			a_next = self.target_actor_net(next_state) + 0.005 * torch.randn_like(action) # modify 
			#a_next = self.target_actor_net(next_state)

			q_next1 = self.target_critic_net1(next_state, a_next)
			q_next2 = self.target_critic_net2(next_state, a_next)# modify 
		 	# select min q value from q_next1 and q_next2 (double Q learning)
			q_target = reward + self.gamma * torch.min(q_next1, q_next2) * (1 - done)# modify 
			#q_target = reward + self.gamma * (q_next1) * (1 - done)

		
		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		critic_loss2 = criterion(q_value2, q_target)# modify 

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_net2.zero_grad()# modify 
		critic_loss2.backward()# modify 
		self.critic_opt2.step()# modify 

		
		 # modify 
		'''
		## Delayed Actor(Policy) Updates ##
		#if self.total_time_step % self.update_freq == 0:
		# 	## update actor ##
		# 	# actor loss
		# 	# select action a from behavior actor network (a is different from sample transition's action)
		# 	# get Q from behavior critic network, mean Q value -> objective function
		# 	# maximize (objective function) = minimize -1 * (objective function)
		action = self.actor_net(state)
		actor_loss = -1 * (self.critic_net1(state, action).mean())
		# 	# optimize actor
		self.actor_net.zero_grad()
		actor_loss.backward()
		self.actor_opt.step()
		'''
		## Delayed Actor(Policy) Updates ##
		if self.total_time_step % self.update_freq == 0:
		# 	## update actor ##
		# 	# actor loss
		# 	# select action a from behavior actor network (a is different from sample transition's action)
		# 	# get Q from behavior critic network, mean Q value -> objective function
		# 	# maximize (objective function) = minimize -1 * (objective function)
			action = self.actor_net(state)
			actor_loss = -1 * (self.critic_net1(state, action).mean())
		# 	# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
