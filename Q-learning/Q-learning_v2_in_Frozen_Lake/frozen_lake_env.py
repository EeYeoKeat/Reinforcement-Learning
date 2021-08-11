# -*- coding: utf-8 -*-
"""
@author: EE

"""
import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd
from Q_learning_agent_v2 import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = Agent(alpha=0.001, gamma=0.9, n_actions=4, n_states=16, eps_start=1.0, eps_min=0.01, eps_dec=0.9999995)
    scores = []
    win_percentage_list = []
    n_games = 500000
    
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            # the info may be replaced with _
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score += reward
            observation = observation_
            
        scores.append(score)
        
        if i % 100 == 0:
            # calculate the average of win percentage for last 100 games
            win_percentage = np.mean(scores[-100:])
            win_percentage_list.append(win_percentage)
            if i % 1000 == 0:
                print('episode',i, 'win percentage %.2f' % win_percentage, 'epsilon %.2f' % agent.epsilon)
                
        
    plt.plot(win_percentage_list)
    plt.show()
            



