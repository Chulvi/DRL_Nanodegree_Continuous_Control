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

The agent has solved the problem after 130 games:

<img src="https://github.com/Chulvi/DRL_Nanodegree_Continuous_Control/blob/main/images/rewards.png" width="800"></img>

```
Game 10  --->  Avg Reward: 0.85  ---> Reward: 0.74
Game 20  --->  Avg Reward: 1.51  ---> Reward: 2.87
Game 30  --->  Avg Reward: 3.94  ---> Reward: 15.8
Game 40  --->  Avg Reward: 7.24  ---> Reward: 20.61
Game 50  --->  Avg Reward: 12.16  ---> Reward: 37.04
Game 60  --->  Avg Reward: 16.28  ---> Reward: 37.48
Game 70  --->  Avg Reward: 19.14  ---> Reward: 35.51
Game 80  --->  Avg Reward: 21.07  ---> Reward: 31.54
Game 90  --->  Avg Reward: 22.03  ---> Reward: 31.25
Game 100  --->  Avg Reward: 23.22  ---> Reward: 32.88
Game 110  --->  Avg Reward: 26.61  ---> Reward: 33.71
Game 120  --->  Avg Reward: 29.85  ---> Reward: 35.77
Game 130  --->  Avg Reward: 32.47  ---> Reward: 34.61
```

### Ideas for Future Work

- Try to implement different algorithms to compare results. (PPO, A3C)
- Experiment with 'All time high' checkpoints to resume avoiding exaggerated dropping.
- Increase number of agents.
