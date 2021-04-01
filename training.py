from setuptools import setup
import gym
import flappy_env
from stable_baselines3 import PPO

def human_playing():
	env = flappy_env.FlappyEnv()
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
	env = flappy_env.FlappyEnv()
	obs = env.reset()
	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=2e6)

	for i in range(1000):
	    action, _states = model.predict(obs, deterministic=True)
	    obs, reward, done, info = env.step(action)
	    if done:
	      obs = env.reset()
	env.close()

ai_playing()