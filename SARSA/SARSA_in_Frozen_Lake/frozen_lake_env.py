# -*- coding: utf-8 -*-
"""
@author: EE

"""
import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd
from SARSA_agent import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = Agent(alpha=1e-5, gamma=0.9, n_actions=4, n_states=16, eps_start=1.0, eps_min=0.1, eps_dec=0.999999)
    scores = []
    total_reward = []
    win_percentage_list = []
    n_games = 100000
    
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        # For SARSA model, the action taken before learn update
        action = agent.choose_action(observation)
        while not done:
            # env.render()
            # the info may be replaced with _
            observation_, reward, done, info = env.step(action)
            
            # In SARSA as on-policy, agent choose next action based on next state
            action_ = agent.choose_action(observation_)
            
            # SARSA will learn from the transition state, action, reward, state_, action_
            agent.learn(observation, action, reward, observation_, action_)
            score += reward
            observation = observation_
            
            # Agent is confirm to choose that action
            action = action_
            
        scores.append(score)
        if i % 100 == 0:
            # calculate the average of win percentage for last 100 games
            win_percentage = np.mean(scores[-100:])
            # win_percentage_list.append(win_percentage)
            if i % 100 == 0:
                print('episode',i, 'win percentage %.2f' % win_percentage, 'epsilon %.2f' % agent.epsilon)
                
        
    plt.plot(win_percentage)
    plt.show()
            



