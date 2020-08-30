# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:07:31 2020

@author: user
"""
import gym
env = gym.make('BipedalWalker-v3')

for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("{} timesteps taken for the episode".format(t+1))
            break
    if done:
        observation = env.reset()
    
env.close()