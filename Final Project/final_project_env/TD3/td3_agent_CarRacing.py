import torch
import torch.nn as nn
import numpy as np
from .base_agent import TD3BaseAgent
from .models.CarRacing_model import ActorNetSimple, CriticNetSimple
from .environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from .base_agent import OUNoiseGenerator, GaussianNoise
import time


class CarRacingTD3Agent(TD3BaseAgent):
    def __init__(self, config):
        super(CarRacingTD3Agent, self).__init__(config)
        # initialize environment
        self.env = CarRacingEnvironment(
            N_frame=4, test=False, scenario=config["scenario"])
        self.test_env = CarRacingEnvironment(
            N_frame=4, test=True, scenario=config["scenario"])
        
        # behavior network
        self.actor_net = ActorNetSimple(
            config["obs_size"], self.env.action_space.shape[0], 4)
        self.critic_net1 = CriticNetSimple(
            config["obs_size"], self.env.action_space.shape[0], 4)
        self.critic_net2 = CriticNetSimple(
            config["obs_size"], self.env.action_space.shape[0], 4)
        self.actor_net.to(self.device)
        self.critic_net1.to(self.device)
        self.critic_net2.to(self.device)
        # target network
        self.target_actor_net = ActorNetSimple(
            config["obs_size"], self.env.action_space.shape[0], 4)
        self.target_critic_net1 = CriticNetSimple(
            config["obs_size"], self.env.action_space.shape[0], 4)
        self.target_critic_net2 = CriticNetSimple(
            config["obs_size"], self.env.action_space.shape[0], 4)
        self.target_actor_net.to(self.device)
        self.target_critic_net1.to(self.device)
        self.target_critic_net2.to(self.device)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
        
        # set optimizer
        self.lra = config["lra"]
        self.lrc = config["lrc"]
        
        self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
        self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
        self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

        # choose Gaussian noise or OU noise

        noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
        noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
        self.noise = OUNoiseGenerator(noise_mean, noise_std)

        # self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
        
        self.scenario = config["scenario"]
        
        self.right_count = 0
        self.left_count = 0
        self.straight_count = 0
        self.action_state = "S"
        
        
    def warmup_action(self, state):
        action = [0.0, 0.0]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.my_noise_2(state, action)
        
        return action

    
    def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
        ### TODO ###
        # based on the behavior (actor) network and exploration noise
        # action = [0.0, 0.0]
        # pre_action = [0.1, 0]
        # count = 0
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor_net(state, brake_rate).cpu().numpy().squeeze()
            action += (self.noise.generate() * sigma) #B4
            # action = self.my_noise_2(state, action)
            # pre_action = action

        # print("action:", action)
        
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
        q_value2 = self.critic_net2(state, action)
        with torch.no_grad():
        # 	# select action a_next from target actor network and add noise for smoothing
            a_next = self.target_actor_net(next_state) + 0.005 * torch.randn_like(action)
            # a_next = self.target_actor_net(next_state) #B2

            q_next1 = self.target_critic_net1(next_state, a_next)
            q_next2 = self.target_critic_net2(next_state, a_next)
        # 	# select min q value from q_next1 and q_next2 (double Q learning)
            q_target = reward + self.gamma * torch.min(q_next1, q_next2) * (1 - done)
            # q_target = reward + self.gamma * q_next1 * (1 - done) #B1
        
          # critic loss function
        criterion = nn.MSELoss()
        critic_loss1 = criterion(q_value1, q_target)
        critic_loss2 = criterion(q_value2, q_target)

        # optimize critic
        self.critic_net1.zero_grad()
        critic_loss1.backward()
        self.critic_opt1.step()

        self.critic_net2.zero_grad()
        critic_loss2.backward()
        self.critic_opt2.step()

        ## Delayed Actor(Policy) Updates ##
        if self.total_time_step % self.update_freq == 0:
            ## update actor ##
            # actor loss
            # select action a from behavior actor network (a is different from sample transition's action)
            # get Q from behavior critic network, mean Q value -> objective function
            # maximize (objective function) = minimize -1 * (objective function)
            # print("state:", state.shape)
            
            action = self.actor_net(state)
            
            rule_action = []
            for i in range(state.shape[0]):
                rule_action.append(self.my_noise_2(state[i].unsqueeze(0), action))
            rule_action = torch.FloatTensor(np.array(rule_action))
            
            # print("action:", action.shape)
            # print("rule_action:", rule_action.shape)
            
            critic_loss = self.critic_net1(state, action).mean()
            rule_loss = criterion(action.to(self.device),
                                  rule_action.to(self.device))
            

            # print("rule_loss:", rule_loss)
            # print("critic_loss:", critic_loss)
            
            # rule_loss: smaller better
            # critic_loss: larger better
            actor_loss = rule_loss - critic_loss
            
            self.writer.add_scalar('Train/critic_loss',
                                   critic_loss, self.total_time_step)
            self.writer.add_scalar('Train/rule_loss',
                                   rule_loss, self.total_time_step)
            self.writer.add_scalar('Train/actor_loss',
                                   actor_loss, self.total_time_step)

            # optimize actor
            self.actor_net.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

        
    def my_noise_1(self, state, action):
        # print("state.shape:", state.shape)
        obs = state[0][3]
        # print("obs:", obs)

        left_wall = 0
        right_wall = 0
        for i in range(len(obs)):
            if obs[i][0] >= 140 and obs[i][0] <= 200:
                left_wall += 1

            if obs[i][-1] >= 140 and obs[i][-1] <= 200:
                right_wall += 1

        # print("left_wall:", left_wall)
        # print("right_wall:", right_wall)

        if self.scenario == "circle_cw_competition_collisionStop":
            action = [1.0, 0.15]

            if left_wall >= len(obs)*0.5:
                action = [1.0, 0.3]

            if right_wall >= len(obs)*0.5:
                action = [1.0, 0.0]
        elif self.scenario == "austria_competition":
            action = [0.2, 0.0]

            if left_wall >= len(obs)*0.4:
                action = [0.1, 0.75]
            elif left_wall >= len(obs)*0.6:
                action = [0.05, 1.0]

            if right_wall >= len(obs)*0.4:
                action = [0.1, -0.75]
            elif right_wall >= len(obs)*0.6:
                action = [0.05, -1.0]

        # demo not command
        # time.sleep(0.1)

        return np.array(action)


    def my_noise_2(self, state, action):
        # print("state:", state.shape)
        obs = state[0][3]
        # print("obs:", obs.shape)

        left_wall = 0
        right_wall = 0
        for i in range(len(obs)):
            if obs[i][0] >= 140 and obs[i][0] <= 200:
                left_wall += 1

            if obs[i][-1] >= 140 and obs[i][-1] <= 200:
                right_wall += 1

        # print("left_wall:", left_wall)
        # print("right_wall:", right_wall)

        if self.scenario == "circle_cw_competition_collisionStop":
            if left_wall > right_wall:
                action = [0.1, min(1, left_wall/right_wall)]
            elif right_wall > left_wall:
                action = [0.1, -min(1, right_wall/left_wall)]
            else:
                action = [0.2, 0.0]
        elif self.scenario == "austria_competition":
            # if left_wall > right_wall:
            #     action = [0.05, min(1, ((left_wall/right_wall)-1)*1.0)]
            # elif right_wall > left_wall:
            #     action = [0.05, -min(1, ((right_wall/left_wall)-1)*1.0)]
            # else:
            #     action = [0.1, 0.0]
            
            # if left_wall > right_wall:
            #     action = [0.01, 1]
            # elif right_wall > left_wall:
            #     action = [0.01, -1]
            # else:
            #     action = [0.1, 0.0]
                
            if left_wall - right_wall > 4:
                action = [0.01, 1.0]
            elif right_wall - left_wall > 4:
                action = [0.01, -1.0]
            else:
                action = [0.1, 0.0]

        # demo not command
        # time.sleep(0.1)

        return np.array(action)


    def my_noise_3(self, state, action):
        # print("state.shape:", state.shape)
        obs = state[0][3]
        # print("obs:", obs)

        left_wall = 0
        right_wall = 0
        for i in range(len(obs)):
            if obs[i][0] >= 140 and obs[i][0] <= 200:
                left_wall += 1

            if obs[i][-1] >= 140 and obs[i][-1] <= 200:
                right_wall += 1

        # print("left_wall:", left_wall)
        # print("right_wall:", right_wall)

        if self.scenario == "circle_cw_competition_collisionStop":
            if left_wall > right_wall:
                action = [0.1, min(1, left_wall/right_wall)]
            elif right_wall > left_wall:
                action = [0.1, -min(1, right_wall/left_wall)]
            else:
                action = [0.2, 0.0]
        elif self.scenario == "austria_competition":
            if left_wall - right_wall > 4:      # right
                self.right_count += 1
                self.left_count = 0
                self.straight_count = 0
            elif right_wall - left_wall > 4:    # left
                self.right_count = 0
                self.left_count += 1
                self.straight_count = 0
            else:   # straight
                self.right_count = 0
                self.left_count = 0
                self.straight_count += 1
            
            if self.right_count >= 1:
                self.action_state = "R"
            if self.left_count >= 2:
                self.action_state = "L"
            if self.straight_count >= 4:
                self.action_state = "S"

            if self.action_state == "R":
                action = [0.01, 1.0]
            if self.action_state == "L":
                action = [0.01, -1.0]
            if self.action_state == "S":
                action = [0.1, 0.0]
                
            print("action_state:", self.action_state)

        # demo not command
        # time.sleep(0.1)

        return np.array(action)


