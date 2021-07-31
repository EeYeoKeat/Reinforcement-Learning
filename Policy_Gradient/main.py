import numpy as np
import matplotlib.pyplot as plt
import gym
from policy_gradient import PolicyGradient_Agent

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.seed(1) 
    agent = PolicyGradient_Agent(alpha=0.01, gamma=0.99, n_actions=env.action_space.n, n_states=env.observation_space.shape[0])
    scores = []
    n_games = 1000
    
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        
        while not done:
            env.render()
            action = agent.choose_action(observation)
            # the info may be replaced with _
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            
            if done:
                agent.learn()
            score += reward
            observation = observation_
            
        scores.append(score)
        
        if i % 100 == 0:
            # calculate the average of win percentage for last 100 games
            win_percentage = np.mean(scores[-100:])
            print('episode',i, 'win percentage %.2f' % win_percentage)
                
        
    plt.plot(scores)
    plt.show()
            


