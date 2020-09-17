import numpy as np
import matplotlib.pyplot as plt
import random
import gym
from gym import wrappers
from collections import deque
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam

class ReplayMemory(object):
    def __init__(self, memory_size, input_dim):
        self.mem_size = memory_size
        self.mem_counter = 0
        
        # memory for state, action, reward, next_state, terminal
        self.state_mem = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dim),dtype=np.float32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.uint8)
    
    def store_memory(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.state_mem[index] =  state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.new_state_mem[index] = next_state
        self.terminal_mem[index] = done
        self.mem_counter +=1
    
    def sample_memory(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards =  self.reward_mem[batch]
        next_states = self.new_state_mem[batch]
        terminal = self.terminal_mem[batch]
        
        return states, actions, rewards, next_states, terminal
    
def build_dqn(lr, n_actions, input_dim, fc1_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',input_shape=(*input_dim,),data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(n_actions))
    
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
    
    return model


class Agent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, replace, input_dims,
                epsilon_decay=0.995, epsilon_min=0.001, mem_size=1000000):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replace = replace
        self.learn_step = 0
        self.memory = ReplayMemory(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 512)
        self.q_next = build_dqn(lr, n_actions, input_dims, 512)

    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_memory(state, action, reward, next_state, done)
        
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            # exploration choose random action
            action = np.random.choice(self.action_space)
        else:
            # exploitation
            state = np.array([observation],copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        
        return action
    
    def learn(self):
        if self.memory.mem_counter > self.batch_size:
            state, action, reward, new_state, done = \
                                    self.memory.sample_memory(self.batch_size)

            self.replace_target_network()

            q_eval = self.q_eval.predict(state)

            q_next = self.q_next.predict(new_state)

            q_target = q_eval[:]
            indices = np.arange(self.batch_size)
            q_target[indices, action] = reward + self.gamma*np.max(q_next, axis=1)*(1 - done)
            self.q_eval.train_on_batch(state, q_target)

            self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
            self.learn_step += 1