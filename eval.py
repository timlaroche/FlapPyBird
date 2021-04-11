def ai_eval():
	env = flappy_env.FlappyEnv(server=False)
	# model = PPO.load("./fixedreward_lr_weightednototjump_newobs_cnn", env=env)
	model = DQN.load("./1e7bwcnndqn", env=env)
	obs = env.reset()
	for i in range(1000):
		action, _state = model.predict(obs, deterministic=True)
		#action = env.action_space.sample()
		#print(action)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			env.reset()