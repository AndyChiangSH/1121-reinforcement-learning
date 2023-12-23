import argparse
from collections import deque
import itertools
import random
import time
import cv2
from matplotlib.pylab import f

# import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from racecar_gym.env import RaceEnv

import gymnasium as gym
from numpy import array, float32


class CarRacingEnvironment:
    def __init__(self, N_frame=4, test=False, scenario='circle_cw_competition_collisionStop'):
        self.env = RaceEnv(
            scenario=scenario,
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False if 'collisionStop' in scenario else True,
            test=test
        )
        
        self.test = test
        # if self.test:
        #     self.env = env
        #     # self.env = gym.wrappers.RecordVideo(self.env, 'video_Gousenoise_reward_B4_2')
        # else:
        #     self.env = env
            
        if scenario == 'circle_cw_competition_collisionStop':
            self.action_space = gym.spaces.box.Box(
                low=0.5, high=1, shape=(2,), dtype=float32)
        else:
            self.action_space = self.env.action_space
        
        self.observation_space = self.env.observation_space
        self.ep_len = 0
        self.frames = deque(maxlen=N_frame)
    
    def check_car_position(self, obs):
        # cut the image to get the part where the car is
        part_image = obs
        # print(part_image.shape)

        road_color_lower = np.array([200, 200, 200], dtype=np.uint8)
        road_color_upper = np.array([250, 250, 250], dtype=np.uint8)
        grass_color_lower = np.array([140, 140, 140], dtype=np.uint8)
        grass_color_upper = np.array([200, 200, 200], dtype=np.uint8)
        road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
        grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)
        # count the number of pixels in the road and grass
        road_pixel_count = cv2.countNonZero(road_mask)
        grass_pixel_count = cv2.countNonZero(grass_mask)

        # save image for debugging
        filename = "images/image" + str(self.ep_len) + ".jpg"
        cv2.imwrite(filename, part_image)

        return road_pixel_count, grass_pixel_count

    def step(self, action):
        obs, reward, terminates, truncates, info = self.env.step(action)
        obs = np.transpose(obs, (1, 2, 0))
        original_reward = reward
        original_terminates = terminates
        self.ep_len += 1
        road_pixel_count, grass_pixel_count = self.check_car_position(obs)
        info["road_pixel_count"] = road_pixel_count
        info["grass_pixel_count"] = grass_pixel_count
        
        # print("reward:", reward)

        # my reward shaping strategy, you can try your own
        # if road_pixel_count < 10:
        # 	terminates = True
        # 	reward = -100
        
        # print("info:", info)
        
        # panalty = 0
        # for p in info['collision_penalties']:
        #     panalty += p
        
        # TD3-circle-1 reward    
        # reward += (info['lap'] + info['progress'] - 1) - panalty*0.1

        # TD3-circle-2 reward
        # reward += (info['lap'] + info['progress'] - 1) - (panalty + info['n_collision'])
  
        # TD3-circle-3 reward
        # reward -= info["grass_pixel_count"] * 0.1

        # TD3-circle-4 reward
        # reward += (info['lap'] + info['progress'] - 1) + info['time'] * 0.1

        # TD3-circle-5 reward
        # reward = (info['lap'] - 1) + info['progress']

        # TD3-circle-6 reward
        # reward = info['progress']

        # TD3-circle-7 reward
        # reward = (info['lap'] - 1) + info['progress'] - info["dist_goal"]

        # TD3-circle-8 reward
        # reward += (info['lap'] - 1) + info['progress'] - \
        #     info["dist_goal"] + info["obstacle"]

        # TD3-circle-9 reward
        # reward -= panalty
        
        # TD3-circle-14 reward
        # if info["wall_collision"]:
        #     terminates = True
        #     reward = reward * 0.9
        
        # TD3-circle-15 reward
        # reward += info["obstacle"] * 0.01

        # reward +=  (0.01 * info['progress'] - 0.1 * info['n_collision']) - 0.01 * info['wrong_way'] + 0.01 * (info['lap']-1) - panalty * 0.01 #reward2
        # reward +=  (-0.01 * info['n_collision']) - 0.01 * info['wrong_way'] + 0.01 * (info['lap']-1) - panalty * 0.001 #reward3
        # reward += 0.01 * info['progress'] #reward/fine_tune

        # at first agent should go more progress as possible, and then keep the car in the road, and then try to avoid collision
        # reward = reward - info['grass_pixel_count'] * 0.0001
        # if info['progress'] < 0.2:
        # 	reward += 0.05 * info['progress'] - 0.01 * info['n_collision'] - 0.1 * info['wrong_way']
        # else:
        # 	reward += 0.05 * info['progress'] - 0.01 * info['n_collision'] - 0.1 * info['wrong_way'] + 0.1 * (info['lap'] - 1) - panalty * 0.001

        # reward function: hope lap + progress as large as possible and no collision
        # reward += 100 * (info['lap'] - 1) + 10 * info['progress'] - 100 * info['n_collision'] - 100 * info['wrong_way']
        # reward += 0.1 * info['progress'] - 0.1 * info['n_collision'] - 0.1 * info['wrong_way'] + 0.1 * (info['lap'] - 1)

        # convert to grayscale
        # obs = np.transpose(obs, (1, 2, 0))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96
        
        # resize image
        obs = cv2.resize(obs, (32, 32), interpolation=cv2.INTER_AREA)

        # save image for debugging
        # filename = "images/image" + str(self.ep_len) + ".jpg"
        # cv2.imwrite(filename, obs)

        # frame stacking
        # print("obs.shape:", obs.shape)
        self.frames.append(obs)
        # print("obs.shape:", obs.shape)
        obs = np.stack(self.frames, axis=0)
        # obs = np.transpose(obs, (2, 0, 1))

        if self.test:
            # enable this line to recover the original reward
            reward = original_reward
            # enable this line to recover the original terminates signal, disable this to accerlate evaluation
            terminates = original_terminates

        return obs, reward, terminates, truncates, info
    
    def reset(self, seed=False):
        if seed is not False:
            obs, info = self.env.reset(seed = seed)
        else:
            obs, info = self.env.reset()
        self.ep_len = 0
        
        # convert to grayscale obs = 128*128*3
        obs = np.transpose(obs, (1, 2, 0))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96
        obs = cv2.resize(obs, (32, 32), interpolation=cv2.INTER_AREA)

        # frame stacking
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)
        # print(f'obs.shape: {obs.shape}')

        return obs, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

if __name__ == '__main__':
    env = CarRacingEnvironment(test=True)
    obs, info = env.reset()
    done = False
    total_reward = 0
    total_length = 0
    t = 0
    road = 0
    grass = 0
    while not done:
        t += 1
        action = env.action_space.sample()
        # action[2] = 0.0
        obs, reward, terminates, truncates, info = env.step(action)
        road_pixel_count, grass_pixel_count = info['road_pixel_count'], info['grass_pixel_count']
        road = road_pixel_count
        grass = grass_pixel_count
        print(f'{t}: reward: {reward}')
        print(f'road: {road}, grass: {grass}')
        total_reward += reward
        total_length += 1
        # env.render()
        if terminates or truncates:
            done = True

    print("Total reward: ", total_reward)
    print("Total length: ", total_length)
    env.close()
