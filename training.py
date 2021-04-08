from setuptools import setup
import gym
import flappy_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor

def human_playing():
	env = flappy_env.FlappyEnv(server=False)
	env.reset()
	env.step(action=[])
	while env.running:
		actions = env.get_actions()
		obs, reward, done, info = env.step(action=actions)
		print(f"""
			obs: {obs}
			reward: {reward}
			done: {done}
			info: {info}
			""")
		env.render()
		env.close()

def ai_playing():
	env = flappy_env.FlappyEnv(server=True)
	env = Monitor(env, "here")
	obs = env.reset()
	model = PPO("CnnPolicy", env, verbose=1)
	model.learn(total_timesteps=1e8)
	model.save("fixedreward_lr_weightednototjump_newobs_cnn")

	# for i in range(1000):
	# 	# action, _state = model.predict(obs, deterministic=True)
	# 	action = env.action_space.sample()
	# 	#print(action)
	# 	obs, reward, done, info = env.step(action)
	# 	env.render()
	# 	if done:
	# 		env.reset()

def ai_eval():
	env = flappy_env.FlappyEnv(server=False)
	model = PPO.load("./fixedreward_lr_weightednototjump_newobs_cnn", env=env)
	obs = env.reset()
	for i in range(1000):
		action, _state = model.predict(obs, deterministic=True)
		#action = env.action_space.sample()
		#print(action)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			env.reset()

ai_playing()