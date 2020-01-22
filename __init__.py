from dqn import DQN
import numpy as np
import gym, cv2
from tqdm import tqdm

env = gym.make("CartPole-v1") # initialize the enviroment
player = DQN(input_shape=env.observation_space.shape, output_shape=env.action_space.n) # initialize the DQN givin the input and output shape
check_point_epoch = [1, 5, 10, 20, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350,
                400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] # checkpoints for saving just the policy net
MODEL_NAME = "second/second" # model name

def train_the_player():
    EPOCHS = 2000
    decay_exp = 0.01
    exploration_rate = 1
    # player.load(MODEL_NAME, load_target_net=True) # load the model both policy and target

    for epoch in tqdm(range(1000, EPOCHS+1), unit="epoch"):
        state = env.reset() # reset the state
        done = False
        while not done:
            # exploration vs exploitation
            if np.random.uniform() > exploration_rate:
                action = player.play(state.reshape(1, 4)) # action from policy net
            else:
                action = env.action_space.sample() # random action

            next_state, reward, done, info = env.step(action) # take the action

            player.update((state.reshape(1, 4), action, reward, next_state.reshape(1, 4), done)) # updating player memory
            sample = player.create_sample(32) # create a random sample from the memory

            # if we have >64 data in memory we can create a sample
            if sample:
                player.train(sample)

            state = next_state # setting the next state as the current state

        exploration_rate = 0.01 + (1 - 0.01) * np.exp(-decay_exp*epoch) # changing exploration rate

        if not epoch % 10:
            player.save(MODEL_NAME, save_target_net=True, verbose=False) # save the model both policy and target without print something

        if epoch in check_point_epoch:
            player.save(f"{MODEL_NAME}_{epoch}", save_target_net=False, verbose=False) # save the model just policy without print something

def playing(model_name, load_target_net=False):
    player.load(model_name, load_target_net=False) # load the model
    state = env.reset() # reset the state
    for t in range(200):
        env.render() # render the game
        action = player.play(state.reshape(1, 4)) # take an action according to the neural network
        state, reward, done, info = env.step(action)
    env.close()

if __name__ == "__main__":
    # train_the_player()
    playing(MODEL_NAME)