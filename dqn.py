from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import random

class DQN():
    # setting input and output shapes to None just in case you want to use the class only for loading the model
    def __init__(self, input_shape=None, output_shape=None, discount=0.99, learning_rate=0.001, memory_size=10000, sample_size=64, update_target_every=50):
        self.input_shape = input_shape # choosing the input shape
        self.output_shape = output_shape # choosing the output shape
        self.discount = discount
        self.learning_rate = learning_rate

        if input_shape and output_shape:
            self.policy_net = self.create_model() # create the policy neural network
            self.target_net = self.policy_net # clone the policy neural network to target neural network

        self.memory = deque(maxlen=memory_size)
        self.sample_size = sample_size
        self.target_counter = 0
        self.update_target_every = update_target_every

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.input_shape, activation="relu")) # fully connected layer
        model.add(Dense(100, activation="relu")) # fully connected layer
        model.add(Dense(self.output_shape, activation="softmax")) # output layer
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate)) # choosing the loss and the optimizer
        return model # return the neural network

    def play(self, state):
        prediction = self.policy_net.predict(state) # take an action according to the policy net
        return np.argmax(prediction) # return the max index

    def train(self, sample):
    
        states = np.array([i[0] for i in sample]) # every state in sample
        next_states = np.array([i[3] for i in sample]) # every next state in sample

        next_qs = self.target_net.predict(next_states) # calcolate the q for every next state in the sample using target net
        next_qs_eval = self.policy_net.predict(next_states) # calcolate the q for every next state in the sample using policy net
        current_qs_pred = self.policy_net.predict(states) # calcolate the q for every state in the sample using policy net

        for index, (state, action, reward, next_state, done) in enumerate(sample):

            if not done:
                target = reward + self.discount * next_qs[index, np.argmax(next_qs_eval[index])] # formula to calcolate the target
            else:
                target = reward # if this is last frame (i.e. done=True)

            current_qs_pred[index, action] = target # update Q(s,a) with the target

        self.policy_net.fit(states, current_qs_pred, verbose=0, batch_size=self.sample_size, shuffle=False)

        self.target_counter += 1 # updating the counter

        # every x amount of time update the target net with the policy net's weights
        if not self.target_counter % self.update_target_every:
            self.target_net.set_weights(self.policy_net.get_weights())

    def update(self, experience):
        self.memory.append(experience) # append experience to memory

    def create_sample(self):
        # return a random sample if we have enough experience in memory
        if len(self.memory) < self.sample_size:
            return None # return none
        else:
            return random.sample(self.memory, self.sample_size) # return a random sample from memory

    # save the model
    def save(self, model_name, save_target_net=True, verbose=False):
        self.policy_net.save(f"models/{model_name}_policy.h5") # save the policy net
        if verbose:
            print(f"policy net saved as {model_name}_policy.h5")
        # if True save the target net too
        if save_target_net:
            self.target_net.save(f"models/{model_name}_target.h5") # save the target net
            if verbose:
                print(f"target net saved as {model_name}_target.h5")

    # load the model
    def load(self, model_name, load_target_net=False, verbose=False):
        self.policy_net = load_model(f"models/{model_name}_policy.h5") # load the policy net
        # if verbose true print
        if verbose:
            print(f"{model_name}_policy.h5 loaded")
        # if True load the target net too
        if load_target_net:
            self.target_net = load_model(f"models/{model_name}_target.h5") # load the target net
            # if verbose true print
            if verbose:
                print(f"{model_name}_target.h5 loaded")