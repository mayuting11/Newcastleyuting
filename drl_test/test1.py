import gym
from run import run
env = gym.make("CartPole-v1", render_mode="rgb_array")
run(env)
env.close()
