from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-6,
		"lrc": 4.5e-6,
		"replay_buffer_capacity": 5000,
		"logdir": 'log/TD-3-demo/',  # modify
		"update_freq": 2,
		"eval_interval": 10,
		"eval_episode": 1,
	}
 
	agent = CarRacingTD3Agent(config)
	#agent.load_and_evaluate('log/CarRacing/td3_test_new_reward_2/model_818209_918.pth') # modify 
	agent.load_and_evaluate(
		'log/TD-3-demo/model_2168037_917.pth')  # modify
	#agent.load_and_evaluate('log/CarRacing/td3_test_new_reward_2/model_674047_897.pth') # modify 
	#agent.load_and_evaluate('model_3783694_917.pth') # modify 
	#agent.train()


