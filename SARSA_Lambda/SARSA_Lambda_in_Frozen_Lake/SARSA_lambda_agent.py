# -*- coding: utf-8 -*-
"""
@author: EE
"""

import numpy as np

class SARSALambdaAgent():
    def __init__(self, alpha, gamma, n_actions, n_states, eps_start, eps_min, eps_dec, ld_rate):
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.min_epsilon = eps_min
        self.eps_dec = eps_dec
        
        self.Q = {}
        
        # initialize q
        self.init_Q()
        
        self.lambda_decay = ld_rate
        self.eligibility_trace = self.Q.copy()


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
        
    def learn(self, state, action, reward, state_, action_):        
        target_Q = reward + self.discount_factor*self.Q[(state_, action_)]
        predict_Q = self.Q[(state, action)]
        
        diff_error = target_Q - predict_Q
        
        self.eligibility_trace[(state, )] = 0
        self.eligibility_trace[(state, action)] = 1


        self.Q[(state, action)] += self.learning_rate * diff_error * self.eligibility_trace[(state, action)]

        self.decrement_epsilon()
        # decay eligibility trace after update
        self.eligibility_trace[(state, action)] *= self.discount_factor * self.lambda_decay