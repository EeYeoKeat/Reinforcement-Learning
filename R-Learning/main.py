# -*- coding: utf-8 -*-
"""
@author: EE

"""
import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd
from Rlearning_agent import RLearning_Agent


if __name__ == "__main__":
    env = gym.make('Roulette-v0')
    
    # the actions suppose included last action which is to not play at all
    # in this experiment, if n_action-1 will remove the last action option so that agent will keep playing
    # if in the case of n_action which included option of leave the table, then agent will tend to choose that
    agent = RLearning_Agent(alpha=0.01, beta=0.001, rho=0, n_actions=env.action_space.n, n_states=env.observation_space.n, eps_start=1.0, eps_min=0.1, eps_dec=0.999)
    scores = []
    n_games = 1000

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        avg_rewards = []
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score += reward
            observation = observation_
            
        scores.append(score)
        
        if i % 100 == 0:
            avg_rewards.append(np.mean(scores[-100:]))
    
        print('episode',i, 'scores %.2f' % np.mean(scores), 'epsilon %.2f' % agent.epsilon)       
                
    plt.plot(scores)
    plt.show()
            
            



