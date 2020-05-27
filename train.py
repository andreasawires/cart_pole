import gym
from tensorboardX import SummaryWriter
from dqn import DQN
import numpy as np

if __name__ == "__main__":
    env = gym.make("CartPole-v0") # initialize the enviroment
    # env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True) # recording video of the agent for every episode
    player = DQN(input_shape=env.observation_space.shape, output_shape=env.action_space.n,
                update_target_every=150) # initialize the DQN givin the input and output shape

    # hyper parameters
    EPOCHS = 500
    decay_exp = 0.01
    exploration_rate = 1
    MODEL_NAME = "test1"
    scores = []

    for epoch in range(EPOCHS):
        score = 0
        state = env.reset()
        done = False
        while not done:
            # exploration vs exploitation
            if np.random.uniform() > exploration_rate:
                action = player.play(state.reshape(1,4)) # action from policy net
            else:
                action = env.action_space.sample() # random action

            next_state, reward, done, info = env.step(action) # taking action in the enviroment
            score += reward # add reward to the total score of the round

            player.update((state, action, reward, next_state, done)) # updating player memory
            sample = player.create_sample() # create a random sample from the memory

            # if we have enough data in memory we can create a sample
            if sample:
                player.train(sample)

            state = next_state # setting the next state as the current state

        # save the model both policy and target every 10 epochs
        if not epoch % 10:
            player.save(MODEL_NAME)

        exploration_rate = 0.01 + (1 - 0.01) * np.exp(-decay_exp*epoch) # changing exploration rate

        scores.append(score) # appending score to scores list
        avg_score = np.mean(scores[max(0, epoch-100):(epoch+1)]) # calcolate avg score
        print("epoch", epoch, "score %.2f" % score, "average score %.2f" % avg_score) # printing epoch, score and average score