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
		"lra": 4.5e-4, #4.5e-5, 7
		"lrc": 4.5e-4, #4.5e-5, 7
		"replay_buffer_capacity": 5000,
		"update_freq": 2, #B3
		"eval_interval": 50,
		"eval_episode": 5,
		"logdir": 'TD3/log/TD3-circle-2',
		"scenario": "circle_cw_competition_collisionStop"
	}

	agent = CarRacingTD3Agent(config)
	# agent.load('./log/CarRacing/td3_test_Gousenoise_reward/model_1416250_909.pth')
	# agent.load('/home/bryant/Documents/112_1/rl/final_project/final_project_env/TD3/model_906689_0.pth')
	# agent.load('/home/bryant/Documents/112_1/rl/final_project/final_project_env/log/CarRacing/final_project/fine_tune/model_30170_26.pth')
	agent.train()


