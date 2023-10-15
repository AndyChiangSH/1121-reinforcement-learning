from dqn_agent_atari import AtariDQNAgent

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_min": 0.1,
        "warmup_steps": 20000,
        # "warmup_steps": 2000, # test
        "eps_decay": 1000000,
        "eval_epsilon": 0.01,
        "replay_buffer_capacity": 100000,
        "logdir": 'log/DQN_6/',
        "update_freq": 4,
        "update_target_freq": 10000,
        "learning_rate": 0.0000625,
        "eval_interval": 100,
        # "eval_interval": 5,    # test
        "eval_episode": 5,
        "env_id": 'ALE/MsPacman-v5',
        "use_double": False,
    }
    agent = AtariDQNAgent(config)
    agent.load_and_evaluate("./log/DQN_6/model_1615054_3038.pth")