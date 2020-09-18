# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:47:27 2020

@author: YK
"""

import numpy as np
from dqn_keras import Agent
from utils import plotLearning, make_env

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    
    epochs = 100
    load_checkpoint = False
    best_score = 0
    agent = Agent(lr=0.01, gamma=0.99, n_actions=6, epsilon=1.0, batch_size=32, replace=100, input_dims=(4,80,80),
                epsilon_decay=0.995, epsilon_min=0.001, mem_size=100)

        
    filename = 'PongNoFrameskip-v4.png'
    
    scores, eps_history = [], []
    n_steps = 0
    
    for i in range(epochs):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            
            env.render()
            
            action =  agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()
            observation = observation_
        
        scores.append(score)
        
        avg_score = np.mean(scores[-10:])
        print('epoch :',i, 'reward score:',score,
              'last 10 average scores:',avg_score,
              'epsilon %.2f' % agent.epsilon, 'steps:', n_steps)
        
        if avg_score > best_score:
            print('Agent get %.2f break best score record %.2f!' %(avg_score, best_score))
            best_score = avg_score
            
        eps_history.append(agent.epsilon)
        
    x = [i+1 for i in range(epochs)]
    
    plot_learning_curve(x, scores, eps_history, filename)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            