# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gym
from drl_test_model.run import run

def drl_test():
    # Use a breakpoint in the code line below to debug your script.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    run(env)
    env.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    drl_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

