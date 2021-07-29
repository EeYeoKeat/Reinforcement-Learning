# -*- coding: utf-8 -*-
"""
@author: EE
"""

import numpy as np

class RLearning_Agent():
    def __init__(self, alpha, beta, rho,  n_actions, n_states, eps_start, eps_min, eps_dec):
        self.learning_rate = alpha
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.min_epsilon = eps_min
        self.eps_dec = eps_dec
        self.rho = rho
        self.beta = beta
        
        self.Q = {}
        # initialize q
        self.init_Q()
        
    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0
                
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)]) #if can, better reindex the action before choose
            action = np.argmax(actions)
        return action        
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.min_epsilon else self.min_epsilon
        
    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)
        
        self.Q[(state, action)] += self.learning_rate * (reward - self.rho + self.Q[(state_,a_max)]-self.Q[(state, action)])
        
        current_actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
        current_a_max = np.argmax(current_actions)
        
        if(self.Q[(state, action)] == self.Q[(state,current_a_max)]):
            self.rho += self.beta*(reward - self.rho + self.Q[(state_,a_max)] - self.Q[(state,current_a_max)])
        
        self.decrement_epsilon()
        
            

