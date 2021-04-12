from setuptools import setup
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import argparse

def train_dqn(itr = 0, timesteps = 1e7, use_dummy_video = True):
	env = flappy_env.FlappyEnv(use_dummy_video)
	env = Monitor(env, f"flappy_dqn_{itr}")
	obs = env.reset()
	model = DQN(
		"CnnPolicy", 
		env, 
		verbose = 1, 
		optimize_memory_usage = True, 
		buffer_size = 500000, 
		learning_rate = 1e-5, 
		tensorboard_log = f"./dqn_flappy_tensorboard_{itr}/")
	model.learn(total_timesteps = timesteps)
	model.save(f"dqn_flappy_{itr}")

def train_ppo(itr=0, timesteps=1e7, use_dummy_video = True):
	env = flappy_env.FlappyEnv(use_dummy_video)
	env = Monitor(env, f"flappy_ppo_{itr}")
	obs = env.reset()
	model = PPO(
		"CnnPolicy", 
		env, 
		verbose=1, 
		learning_rate=1e-5,
		tensorboard_log = f"./ppo_flappy_tensorboard_{itr}/")
	model.learn(total_timesteps = timesteps)
	model.save(f"ppo_flappy_{itr}")

def ultimate_training():
	for i in range(3):
		train_dqn(i)
	for i in range(3):
		train_ppo(i)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--algorithm", 
		choices=['ppo', 'dqn', 'ultimate'],
		help = "The RL algorithm to use: PPO, DQN, Ultimate (train both 3 times) })",
		required = True
		)
	args = parser.parse_args()
	import flappy_env # Import here so argparse loads faster
	if args.algorithm == "ppo":
		train_ppo(0) 
	elif args.algorithm == "dqn":
		train_dqn(0)
	elif args.algorithm == "ultimate":
		ultimate_training()

