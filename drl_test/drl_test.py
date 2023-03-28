import gym
from drl_test_model.run import run
env = gym.make("CartPole-v1", render_mode="rgb_array")
run(env)
env.close()
