### Learning Algorithm
To resolve the challenge, has been used the algorithm **Dueling Double Deep Q Learning**. A combination of two improvements to the basic DQN:

Dueling DQN | Double DQN
------------ | -------------
![Image of dueling](https://miro.medium.com/max/1468/1*81-seZY1rVwC0wzXBprFJg.png) | ![Image of double](https://cdn-media-1.freecodecamp.org/images/1*g5l4q162gDRZAAsFWtX7Nw.png)

The **deep neural network** uses two hidden layers (64 units and 128 units). The **parameters** below have had the most successful results for this algorithm:

```python
BUFFER_SIZE = int(100000)  # replay buffer size
BATCH_SIZE = 64            # minibatch size
GAMMA = 0.99               # discount factor
TAU = 0.001                # for soft update of target parameters
LR = 0.0005                # learning rate 
UPDATE_EVERY = 4           # how often to update the network
LAYERS = [64, 128]         # neural network num of layers

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
N_GAMES = 2000
```


### Results

The agent has solved the problem after 400 games, improving the score during 2000 games:

<img src="https://github.com/Chulvi/DRL_Nanodegree_Navigation/blob/main/images/rewards.png" width="800"></img>

```
Game 100  --->  Avg Reward: 2.05
Game 200  --->  Avg Reward: 7.22
Game 300  --->  Avg Reward: 11.78
Game 400  --->  Avg Reward: 13.11
Game 500  --->  Avg Reward: 15.08
Game 600  --->  Avg Reward: 16.04
Game 700  --->  Avg Reward: 15.78
Game 800  --->  Avg Reward: 17.0
Game 900  --->  Avg Reward: 15.84
Game 1000  --->  Avg Reward: 15.73
Game 1100  --->  Avg Reward: 15.4
Game 1200  --->  Avg Reward: 16.53
Game 1300  --->  Avg Reward: 16.61
Game 1400  --->  Avg Reward: 15.81
Game 1500  --->  Avg Reward: 16.51
Game 1600  --->  Avg Reward: 16.12
Game 1700  --->  Avg Reward: 16.83
Game 1800  --->  Avg Reward: 16.42
Game 1900  --->  Avg Reward: 16.16
```

### Ideas for Future Work

- Try to implement the ***Rainbow*** paper (mix of several DQN improvements)
- Implement a ***Dynamic Difficulty Adjusting*** approach
  - *First step*: learn to pick up all the bananas
  - *Second step*: learn to avoid blue bananas
