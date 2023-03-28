from display_frames_as_gif import display_frames_as_gif


def run(env):
    frames = []
    for i_episode in range(5):
        observation = env.reset()
        for t in range(20):
            frames.append(env.render())
            action = env.action_space.sample()
            observation, reward, done, info, _ = env.step(action)
            if done:
                break
    display_frames_as_gif(frames)
