import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model_4 import AtariNetDQN
import gym
import random

class AtariDQNAgent(DQNBaseAgent):
    def __init__(self, config):
        super(AtariDQNAgent, self).__init__(config)
        ### TODO ###
        # initialize env
        # self.env = ???
        self.env = gym.make(config["env_id"], obs_type=config["obs_type"])

        ### TODO ###
        # initialize test_env
        # self.test_env = ???
        self.test_env = gym.make(config["env_id"], render_mode='human', obs_type=config["obs_type"])

        # initialize behavior network and target network
        self.behavior_net = AtariNetDQN(num_classes=self.env.action_space.n)
        self.behavior_net.to(self.device)
        self.target_net = AtariNetDQN(self.env.action_space.n)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
        
    def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
        ### TODO ###
        # get action from behavior net, with epsilon-greedy selection
        
        # if random.random() < epsilon:
        # 	action = ???
        # else:
        # 	action = ???

        # return action

        # return NotImplementedError
        # print("observation.shape:", observation.shape)
        observation = np.expand_dims(observation, axis=0)
        observation = torch.from_numpy(observation)
        observation = observation.to(self.device, dtype=torch.float32)
        # print("observation.shape:", observation.shape)
        if random.random() < epsilon:
            # action = np.random.randint(0, action_space, size=observation.shape[0])
            action = np.random.randint(0, action_space.n)
        else:
            action = self.behavior_net(observation).argmax(dim=1).cpu().numpy()[0]
            
        # print("action:", action)
        return action
    
    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        ### TODO ###
        # calculate the loss and update the behavior network
        # 1. get Q(s,a) from behavior net
        # 2. get max_a Q(s',a) from target net
        # 3. calculate Q_target = r + gamma * max_a Q(s',a)
        # 4. calculate loss between Q(s,a) and Q_target
        # 5. update behavior net

        
        # q_value = ???
        # with torch.no_grad():
            # q_next = ???

            # if episode terminates at next_state, then q_target = reward
            # q_target = ???

        action = action.type(torch.long)
        # print("action:", action)
        # print("state:", state)
        # print("state.shape:", state.shape)
        q_value = self.behavior_net(state).gather(1, action)
        # print("q_value:", q_value)
        with torch.no_grad():
            if self.use_double:
                q_next = self.behavior_net(next_state)
                action_index = q_next.max(dim=1)[1].view(-1, 1)
                # choose related Q from target net
                q_next = self.target_net(next_state).gather(dim=1, index=action_index.long())
            else:
                q_next = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)

            # if episode terminates at next_state, then q_target = reward
            q_target = reward + self.gamma * q_next * (1 - done)
        
        # criterion = ???
        # loss = criterion(q_value, q_target)
        
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)

        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    

