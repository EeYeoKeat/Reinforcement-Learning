# SARSA agent in Taxi-v3

Test environment is <a href="https://gym.openai.com/envs/Taxi-v3/">Taxi-v3</a> from OpenAIgym.

```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```


* Total 4 locations (labeled by different letters) 
* Task is to pick up the passenger at one location and drop off in another. 
* +20 points for a successful dropoff
* -1 point for every timestep it takes
* -10 point penalty for illegal pick-up and drop-off actions
* In overall, the agent should maximize the reward which from negative number towards zero or positive number.

