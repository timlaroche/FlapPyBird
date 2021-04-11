from setuptools import setup
import gym
import flappy_env

def human_playing():
	# Lets a human play but wrapping around the actual gym environment
	env = flappy_env.FlappyEnv(server=False)
	env.reset()
	env.step(action=[])
	total_reward = 0
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
		total_reward += reward
	env.close()
	print(f"total_reward: {total_reward}")

if __name__ == '__main__':
    human_playing()