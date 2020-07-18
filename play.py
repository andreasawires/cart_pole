import gym
from dqn import DQN

if __name__ == "__main__":
    player = DQN()
    MODEL_NAME = "best_model/best_model"
    player.load(MODEL_NAME)

    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, "recording/play", force=True, video_callable=lambda episode_id: True) # recording video of the agent for every episode
    state = env.reset()
    score = 0
    for t in range(500):
        env.render()
        # action = env.action_space.sample()
        action = player.play(state.reshape(1, 4))
        state, reward, done, info = env.step(action)
        score += reward

    env.close()
    print(f"score: {score}")