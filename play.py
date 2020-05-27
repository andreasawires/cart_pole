import gym
from dqn import DQN

if __name__ == "__main__":
    player = DQN()
    MODEL_NAME = "second/second"
    player.load(MODEL_NAME)

    env = gym.make("CartPole-v0")
    state = env.reset()
    for t in range(500):
        env.render()
        action = player.play(state.reshape(1, 4))
        state, reward, done, info = env.step(action)

    env.close()