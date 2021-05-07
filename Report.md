### Learning Algorithm
To resolve the challenge, has been used the algorithm **Multi-Agent Deep Deterministic Policy Gradient**. This method implement DDPG to be used for multiple agents at the same time, recording all their experiences in the replay buffer:

DDPG | MADDPG
------------ | -------------
![Image of DDPG](https://miro.medium.com/max/1084/1*BVST6rlxL2csw3vxpeBS8Q.png) | ![Image of MADDPG](https://programmersought.com/images/862/5709e3323ebc72a6499d52623798369e.png)

The **deep neural networks** (actor and critic) use two hidden layers (128 units and 128 units) accompanied with batch normalization and dropout. 


The **parameters** below have had the most successful results for this algorithm:

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-3         # learning rate of the actor 
LR_CRITIC = 2e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

ACTOR_UNITS_l1 = 128    # DNN layers units
ACTOR_UNITS_l2 = 128
CRITIC_UNITS_l1 = 128
CRITIC_UNITS_l2 = 128

GAMES = 300
MAX_T = 1000
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

- Try to implement different algorithms to compare results. (PPO, A3C)
- Experiment with 'All time high' checkpoints to resume avoiding exaggerated dropping.
- Increase number of agents.
