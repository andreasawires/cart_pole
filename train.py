import gym
from tensorboardX import SummaryWriter
from dqn import DQN
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    EPOCHS = 200
    learning_rate = 0.001 # learning rate
    discount = 0.97 # discount (aka gamma) used in the formula to calculate target value
    update_target_every = 200 # copy weight to target net every x amount of epochs

    # hyper parameters for exploration vs exploitation 
    decay_exp = 0.001
    exploration_rate = 1
    min_exp_rate = 0.01

    MODEL_NAME = "test"
    scores = []
    best_score = 0

    env = gym.make("CartPole-v0") # initialize the enviroment
    env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True) # recording video of the agent for every episode
    player = DQN(
                    input_shape=env.observation_space.shape,
                    output_shape=env.action_space.n,
                    discount=discount,
                    learning_rate=learning_rate,
                    sample_size=32,
                    update_target_every=update_target_every
                ) # initialize the DQN

    # player.load(MODEL_NAME, load_target_net=True)

    writer = SummaryWriter(f"runs/{MODEL_NAME}") # initialize writer for the graph
    writer.add_hparams({"learning rate": learning_rate, "discount": discount, "target update": update_target_every}, {}, MODEL_NAME) # adding hyper parameters

    for epoch in range(EPOCHS):
        score = 0
        state = env.reset()
        done = False
        while not done:
            # exploration vs exploitation
            if np.random.uniform() > exploration_rate:
                action = player.play(state.reshape(1, 4)) # action from policy net
                action_taken = "POLICY" # saving in a variable the type of action taken to then print it
            else:
                action = env.action_space.sample() # random action
                action_taken = "RANDOM" # saving in a variable the type of action taken to then print it

            next_state, reward, done, info = env.step(action) # taking action in the enviroment
            score += reward # add reward to the total score of the round

            player.update((state, action, reward, next_state, done)) # updating player memory
            sample = player.create_sample() # create a random sample from the memory

            # if we have enough data in memory we can create a sample
            if sample:
                player.train(sample)

            state = next_state # setting the next state as the current state

        if not epoch % 10:
            player.save(MODEL_NAME) # save the model both policy and target every 10 epochs

        exploration_rate = min_exp_rate + (exploration_rate - min_exp_rate) * np.exp(-decay_exp*epoch) # changing exploration rate

        scores.append(score) # appending score to scores list
        avg_score = np.mean(scores[max(0, epoch-100):(epoch+1)]) # calcolate avg score
        
        writer.add_scalar("score", score, epoch) # appending score to the graph
        writer.add_scalar("average score", avg_score, epoch) # appending average score to the graph

        if score > best_score:
            best_score = score # update the best score
            player.save(MODEL_NAME) # save the model both policy and target net every time it beats the score 

        # printing epoch, type of action taken, score, average score, best score
        print("epoch: ", epoch, ", ", action_taken, ", score: %.2f" % score, ", average score: %.2f" % avg_score, 
            ", best score: %.2f" % best_score, sep="") 

    player.save(MODEL_NAME) # save the model both policy and target