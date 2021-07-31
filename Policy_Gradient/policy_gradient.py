import numpy as np
import math
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class PolicyGradient_Agent():
    def __init__(self, alpha, gamma, n_actions, n_states):
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.episode_obs = []
        self.episode_act = []
        self.episode_r = []
        self.model = self.build_NN()
    
    def build_NN(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.n_states)))
        model.add(Dense(64, activation='relu', kernel_initializer=keras.initializers.he_normal()))
        model.add(Dense(64, activation='relu', kernel_initializer=keras.initializers.he_normal()))
        model.add(Dense(self.n_actions, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=self.learning_rate))
        return model
                
    def choose_action(self, state):
        out_prob = self.model(np.array([state]).reshape((1, -1)))
        print(out_prob)
        action = np.random.choice(self.n_actions, p=out_prob.numpy()[0])
        print(action)
        
        return action
    
    def get_action(network, state, num_actions):
        softmax_out = network(state.reshape((1, -1)))
        selected_action = np.random.choice(num_actions, p=softmax_out.numpy()[0])
        return selected_action
    
    def store_transition(self, observation, action, reward):
        self.episode_obs.append(observation)
        self.episode_act.append(action)
        self.episode_r.append(reward)   
    
    def learn(self):    
        reward_sum = 0
        discounted_rewards = []
        for reward in self.episode_r[::-1]:  # reverse buffer r
            reward_sum = self.discount_factor * reward_sum + reward
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        episode_obs = np.vstack(self.episode_obs)
        print('shape of states ', np.shape(episode_obs))
        print('shape of reward ', np.shape(discounted_rewards))
        print('reward ', discounted_rewards)
        target_actions = np.array([[1 if a==i else 0 for i in range(self.n_actions)]  for a in self.episode_act])
        self.model.train_on_batch(episode_obs, target_actions, sample_weight=discounted_rewards)
        

