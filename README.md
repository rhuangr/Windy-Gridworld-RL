# Windy Gridworld using SARSA
This project is an implementation of the algorithm provided in the book ***Reinforcement Learning An Introduction by Barto and Sutton, Exercise 6.5.*** 

An agent is set in a windy gridworld shown below. Its goal is to navigate from the starting position to the end in the least amount of turns possible, using SARSA to calculate state action pairs.
Numbers at the bottom of the grid represent the *wind strength*. Actions taken from a column with a non-zero wind strength will push the agent up by that amount (unless already at the edge of the gridworld).

![image](https://github.com/rhuangr/Windy-Gridworld-RL/assets/170949635/2e9074e1-11b4-4b2a-9b2f-16710cb95b1b)

For this exercise, all transitions from one state to another will generate a reward of -1, besides the state action pair leading the agent to the end, which will generate a reward of 0.
Assuming correct implementation, all state action pairs should have updated values corresponding to the number of moves needed to reach the goal but negative.
The picture above shows the ideal path for a 7x10 gridword, with starting point *S* and goal *G*.

## Installation
1. Clone the repository
2. Make sure numpy is installed
3. Run the file. *While in the cloned project's directory, paste this command into the terminal `python WindyGridworld.py`* :thumbsup:

## Notes
You can play around by changing the starting position and end goal of the exercise. You can also change the dimensions of the gridworld along with the wind strength and how much the agent trains.
Shown below are the variables you should change if you wish to modify the exercise. This snippet will be located at the bottom of the file 😸

```
start= [3,0]
end  = [3,7]

# def GridWorld(#rows, #columns, start, end)
world  =  GridWorld(7, 10 , start, end)

# key: index of the column, value: strength of the wind
windStrength  = {3:1, 4:1, 5:1, 6:2, 7:2, 8:1}
world.setWind(windStrength)

agent  =  Agent(world)
agent.train(400)

```
