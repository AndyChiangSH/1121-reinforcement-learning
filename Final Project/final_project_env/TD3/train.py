from TD3.td3_agent_CarRacing import CarRacingTD3Agent
import os
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    # config = {
    #     "gpu": True,
    #     "training_steps": 1e8,
    #     "gamma": 0.99,
    #     "tau": 0.005,
    #     "batch_size": 32,
    #     "warmup_steps": 1000,
    #     "total_episode": 100000,
    #     "lra": 4.5e-5,  # 4.5e-5, 7
    #     "lrc": 4.5e-5,  # 4.5e-5, 7
    #     "replay_buffer_capacity": 10000,
    #     "update_freq": 2, #B3
    #     "eval_interval": 100,
    #     "eval_episode": 10,
    #     "logdir": 'TD3/log/TD3-circle-27',
    #     "scenario": "circle_cw_competition_collisionStop",
    #     "obs_size": 128,
    # }

    config = {
        "gpu": True,
        "training_steps": 1e8,
        "total_episode": 100000,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 32,
        "warmup_steps": 1000,
        "lra": 4.5e-5,  # 4.5e-5, 7
        "lrc": 4.5e-5,  # 4.5e-5, 7
        "replay_buffer_capacity": 10000,
        "update_freq": 2,  # B3
        "eval_interval": 100,
        "eval_episode": 10,
        "logdir": 'TD3/log/TD3-austria-8',
        "scenario": "austria_competition",
        "obs_size": 128,
    }

    print(f"Start training {config['logdir']}...")
    output_folder_path = config['logdir']
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    config_path = f"{output_folder_path}/config.json"
    print(f"Save config to {config_path}")
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    agent = CarRacingTD3Agent(config)
    # agent.load('TD3/log/TD3-circle-15/model_967586_78.pth')
    agent.train()


