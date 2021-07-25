
import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9999999)
parser.add_argument('--eps_min', type=float, default=0.1)
parser.add_argument('--num_games', type=float, default=10000)

args = parser.parse_args()

class SARSA_Agent():
    def __init__(self, alpha, gamma, n_actions, n_states, eps_start, eps_min, eps_dec):
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
     
        self.Q[(state, action)] += self.learning_rate * (reward + self.discount_factor*self.Q[(state_, action_)]-self.Q[(state, action)])
        
        self.decrement_epsilon()

if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    agent = SARSA_Agent(alpha=args.lr, gamma=args.gamma, n_actions=env.nA, n_states=env.nS, eps_start=args.eps, eps_min=args.eps_min, eps_dec=args.eps_decay)
    scores = []
    win_percentage_list = []
    n_games = args.num_games
    
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0

        action = agent.choose_action(observation)
        while not done:
            #env.render()
            # the info may be replaced with _
            observation_, reward, done, info = env.step(action)
            
            # In SARSA as on-policy, agent choose next action based on next state
            action_ = agent.choose_action(observation)
            
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
            win_percentage_list.append(win_percentage)
            if i % 1000 == 0:
                print('episode',i, 'win percentage %.2f' % win_percentage, 'epsilon %.2f' % agent.epsilon)
                
        
    plt.plot(win_percentage_list)
    plt.show()
            



