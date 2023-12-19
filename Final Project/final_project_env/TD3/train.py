from TD3.td3_agent_CarRacing import CarRacingTD3Agent
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 500,
		"total_episode": 100000,
		"lra": 4.5e-5,  # 4.5e-5, 7
		"lrc": 4.5e-5,  # 4.5e-5, 7
		"replay_buffer_capacity": 5000,
		"update_freq": 2, #B3
		"eval_interval": 100,
		"eval_episode": 10,
		"logdir": 'TD3/log/TD3-circle-20',
		"scenario": "circle_cw_competition_collisionStop"
	}

	# config = {
	# 	"gpu": True,
	# 	"training_steps": 1e8,
	# 	"gamma": 0.99,
	# 	"tau": 0.005,
	# 	"batch_size": 32,
	# 	"warmup_steps": 500,
	# 	"total_episode": 100000,
	# 	"lra": 4.5e-6,  # 4.5e-5, 7
	# 	"lrc": 4.5e-6,  # 4.5e-5, 7
	# 	"replay_buffer_capacity": 5000,
	# 	"update_freq": 2,  # B3
	# 	"eval_interval": 100,
	# 	"eval_episode": 10,
	# 	"logdir": 'TD3/log/TD3-circle-15-2',
	# 	"scenario": "circle_cw_competition_collisionStop"
	# }

	agent = CarRacingTD3Agent(config)
	# agent.load('TD3/log/TD3-circle-15/model_967586_78.pth')
	agent.train()


