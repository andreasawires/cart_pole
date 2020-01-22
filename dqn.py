from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
from collections import deque
import random

class DQN():
    def __init__(self, input_shape, output_shape, discount=0.99, update_target_every=10, memory_size=2000):
        self.input_shape = input_shape # choosing the input shape
        self.output_shape = output_shape # choosing the output shape
        self.discount = discount
        self.policy_net = self.create_model() # create the policy neural network
        self.target_net = self.policy_net # clone the policy neural network to target neural network
        self.memory = deque(maxlen=memory_size)
        self.target_counter = 0
        self.update_target_every = update_target_every

    def create_model(self):
        model = Sequential()
        model.add(Dense(20, input_shape=self.input_shape, activation="relu")) # fully connected layer
        model.add(Dense(20, activation="relu")) # fully connected layer
        model.add(Dense(self.output_shape, activation="softmax")) # output layer
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"]) # choosing the loss and the optimizer
        return model # return the neural network

    def play(self, state):
        prediction = self.policy_net.predict(state) # take an action according to the policy net
        return np.argmax(prediction)

    def train(self, sample):
        for index, (current_state, action, reward, next_state, done) in enumerate(sample):
            current_q = self.policy_net.predict(current_state)
            next_q = self.target_net.predict(next_state)

            if not done:
                target = reward + self.discount * np.max(next_q)
            else:
                target = reward
            
            current_q[0][action] = target

            self.policy_net.fit(current_state, current_q, epochs=1, verbose=0, shuffle=False)

        self.target_counter += 1 # updating the counter

        # every x amount of time update the target net with the policy net's weights
        if not self.target_counter % self.update_target_every:
            self.target_net.set_weights(self.policy_net.get_weights())

    def update(self, experience):
        self.memory.append(experience)
    
    def create_sample(self, sample_size):
        if len(self.memory) < sample_size:
            return
        else:
            return random.sample(self.memory, sample_size)

    # save the model
    def save(self, model_name, save_target_net=False, verbose=True):
        self.policy_net.save(f"models/{model_name}_policy.h5") # save the policy net
        if verbose:
            print(f"policy net saved as {model_name}_policy.h5")
        # if True save the target net too
        if save_target_net:
            self.target_net.save(f"models/{model_name}_target.h5") # save the target net
            if verbose:
                print(f"target net saved as {model_name}_target.h5")
    
    # load the model
    def load(self, model_name, load_target_net=False, verbose=True):
        self.policy_net = load_model(f"models/{model_name}_policy.h5") # load the policy net
        if verbose:
            print(f"{model_name}_policy.h5 loaded")
        # if True load the target net too
        if load_target_net:
            self.target_net = load_model(f"models/{model_name}_target.h5") # load the target net
            if verbose:
                print(f"{model_name}_target.h5 loaded")