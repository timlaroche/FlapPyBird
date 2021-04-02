from setuptools import setup
import gym
import flappy_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


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
	obs = env.reset()
	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=3e6)
	model.save("fixedreward")

	for i in range(1000):
		# action, _state = model.predict(obs, deterministic=True)
		action = env.action_space.sample()
		print(action)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			env.reset()

ai_playing()