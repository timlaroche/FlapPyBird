from setuptools import setup
import gym
import flappy_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def train_dqn(itr):
	env = flappy_env.FlappyEnv(server=True)
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
	model.learn(total_timesteps = 1e7)
	model.save(f"dqn_flappy_{itr}")

def train_ppo(itr):
	env = flappy_env.FlappyEnv(server=True)
	env = Monitor(env, f"flappy_ppo_{itr}")
	obs = env.reset()
	model = PPO(
		"CnnPolicy", 
		env, 
		verbose=1, 
		learning_rate=1e-5,
		tensorboard_log = f"./ppo_flappy_tensorboard_{itr}/")
	model.learn(total_timesteps=1e7)
	model.save(f"ppo_flappy_{itr}")

def ultimate_training():
	for i in range(3):
		train_dqn(i)
	for i in range(3):
		train_ppo(i)

if __name__ == '__main__':
    ultimate_training()
