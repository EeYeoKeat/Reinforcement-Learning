# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:42:14 2020

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd
import argparse
import time
import os

parser = argparse.ArgumentParser(description='Hyperparameter for R-Learning agent')
parser.add_argument('--lr', type=float, help="learning rate", default=1e-4)
parser.add_argument('--rho', type=float, help="initial average expected reward policy", default=0)
parser.add_argument('--beta', type=float, help="beta", default=1e-5)
parser.add_argument('--eps', type=float, help="initial epsilon", default=1.0)
parser.add_argument('--eps_decay', type=float, help="epsilon decay rate", default=0.999999)
parser.add_argument('--eps_min', type=float, help="minimum epsilon", default=0.1)
parser.add_argument('--num_games', type=float, help="train on how many games?", default=20000) #20000
parser.add_argument('-v', '--verbose', type=int, default=2, help="print out episode")

args = parser.parse_args()

class Rlearning_Agent():
    def __init__(self, alpha, rho, beta, n_actions, n_states, eps_start, eps_min, eps_dec):
        self.learning_rate = alpha
        self.rho = rho
        self.beta = beta
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.min_epsilon = eps_min
        self.eps_dec = eps_dec
        
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

def plot_result(r, avg_r):
    plt.plot(r, color='b')    
    plt.plot(avg_r, color='r')
    plt.title('Scores obtained by R-Learning agent across the games')
    plt.ylabel('Score')
    plt.xlabel('Games')
    plt.legend(['Scores per game', 'Average Scores'], loc='upper left')
    plt.show()
    
def export_result(avg_r):
    if os.path.exists('./agent_scores.csv'):
        print('csv file exists')
        result = pd.read_csv('agent_scores.csv')
        result['R-Learning'] = pd.Series(avg_r)
        result.to_csv('agent_scores.csv', index=False)
        print('results saved!')
    else:
        print('csv file not exists, creating new file.')
        column = {'R-Learning':pd.Series(avg_r)}
        table = pd.DataFrame(column)
        table.to_csv('agent_scores.csv')
        print('results saved!')

if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    agent = Rlearning_Agent(alpha=args.lr, rho=args.rho, beta=args.beta, n_actions=env.nA, n_states=env.nS, eps_start=args.eps, eps_min=args.eps_min, eps_dec=args.eps_decay)
    scores = []
    avg_scores_list = []
    n_games = args.num_games
    start_time = time.time()
    
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            #env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score += reward
            observation = observation_
            
        scores.append(score)
        
        avg_scores = np.mean(scores[-100:])
        avg_scores_list.append(avg_scores)
        
        if i % 100 == 0:
            if args.verbose == 2:
                print('episode',i, 'average scores on last 100 games %.2f' % avg_scores, 'epsilon %.2f' % agent.epsilon)
            elif args.verbose == 1:
                print('Training episode',i, 'win percentage %.2f' % avg_scores)
            else:
                print('Overall Average reward: ',np.mean(avg_scores))
            
                
    end_time = time.time()
    duration = end_time - start_time
    plot_result(scores, avg_scores_list)
    print('Overall mean reward: ',np.mean(scores))
    print('Time Taken: ',duration)
    #export_result(avg_scores_list)

            

