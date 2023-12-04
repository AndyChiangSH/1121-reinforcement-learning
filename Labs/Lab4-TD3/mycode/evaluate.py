from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    # config = {
    # 	"gpu": True,
    # 	"training_steps": 1e8,
    # 	"gamma": 0.99,
    # 	"tau": 0.005,
    # 	"batch_size": 32,
    # 	"warmup_steps": 1000,
    # 	"total_episode": 100000,
    # 	"lra": 4.5e-5,
    # 	"lrc": 4.5e-5,
    # 	"replay_buffer_capacity": 5000,
    # 	"logdir": 'log/TD3-1',
    # 	"update_freq": 2,
    # 	"eval_interval": 10,
    # 	"eval_episode": 10,
    # 	"render": False,
    # }
 
    # config = {
    # 	"gpu": True,
    # 	"training_steps": 1e8,
    # 	"gamma": 0.99,
    # 	"tau": 0.005,
    # 	"batch_size": 32,
    # 	"warmup_steps": 1000,
    # 	"total_episode": 100000,
    # 	"lra": 4.5e-5,
    # 	"lrc": 4.5e-5,
    # 	"replay_buffer_capacity": 5000,
    # 	"logdir": 'log/TD3-2',
    # 	"update_freq": 3,
    # 	"eval_interval": 100,
    # 	"eval_episode": 10,
    # 	"render": False,
    # }
 
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 32,
        "warmup_steps": 1000,
        "total_episode": 100000,
        "lra": 4.5e-5,
        "lrc": 4.5e-5,
        "replay_buffer_capacity": 5000,
        "logdir": 'log/TD3-10',
        "update_freq": 4,
        "eval_interval": 100,
        "eval_episode": 10,
        "render": True,
    }
     
    agent = CarRacingTD3Agent(config)
    agent.load_and_evaluate(
        "log/TD3-9-2/model_4502673_862.pth")


