import argparse
import json
import numpy as np
import requests
from torch import rand
# from stable_baselines3 import TD3
from TD3.td3_agent_CarRacing import CarRacingTD3Agent
import cv2
from collections import deque

frames = deque(maxlen=4)

def connect(agent, url: str = 'http://localhost:5000', first_call = 1):
    step = 1
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)
    
        # obs 
        obs = np.transpose(obs, (1, 2, 0))
        # print(obs.shape)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # print(obs.shape)
        # obs = cv2.resize(obs, (32, 32), interpolation=cv2.INTER_AREA)


        if first_call == 1:
            for _ in range(4):
                frames.append(obs)
            first_call = 0
            obs = np.stack(frames, axis=0)
            # print(f'first: {obs.shape}')
        else:
            frames.append(obs)
            obs = np.stack(frames, axis=0)
        # print(f'final: {obs.shape}')

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.decide_agent_actions(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        # print("result:", result)
        print("Step:", step)
        step += 1
        
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return
        
# scenario = 'austria_competition'
# env = RaceEnv(scenario=scenario,
#         render_mode='rgb_array_birds_eye',
#         reset_when_collision=True if 'austria' in scenario else False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='/localhost:5000', help='The url of the server.')
    args = parser.parse_args()

    # Initialize the RL Agent
    import gymnasium as gym
    
    # my hyperparameters, you can change it as you like
    config = {
        "gpu": True,
        "training_steps": 100000000.0,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 32,
        "warmup_steps": 1000,
        "total_episode": 100000,
        "lra": 4.5e-05,
        "lrc": 4.5e-05,
        "replay_buffer_capacity": 10000,
        "update_freq": 2,
        "eval_interval": 50,
        "eval_episode": 5,
        "logdir": "TD3/log/TD3-circle-27",
        "scenario": "circle_cw_competition_collisionStop",
        "obs_size": 128
    }

    rand_agent = CarRacingTD3Agent(config)
    rand_agent.load(
        'TD3/log/TD3-circle-27/model_398568_62.pth')
    
    connect(rand_agent, url=args.url, first_call=1)
