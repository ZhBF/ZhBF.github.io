# ZhBF的个人博客
Welcome to ZhBF's GitHub Pages

## 强化学习
### Q-Learng算法
#### 实践：在一维离散世界中寻找宝藏
```python

######################################################
# 使用Q-Learing算法创建智能体在一维离散世界中找到宝藏
# 世界：o--------#  其中o为智能体，#为宝藏
######################################################

import numpy as np
import pandas as pd
import random
import time

# Q-learing 智能体
class Agent(object):
    
    def __init__(self, states=(), actions=(), alpha=0.8, gamma=0.8, epsilon=0.8):
        self.states = states
        self.actions = actions
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(np.zeros((len(states),len(actions))),states, actions)
        
    def choose_action(self, state):
        if(random.random()<self.epsilon):
            max_actions = []
            max_q = self.q_table.loc[state].max()
            for action in self.actions:
                if self.q_table.loc[state][action] == max_q:
                    max_actions.append(action)
            action = random.choice(max_actions)
        else:
            action = random.choice(self.actions)
        return action
        
    def update(self, state, action, reward, state_next):
        temp = self.alpha * (reward + self.gamma * self.q_table.loc[state_next].max() - self.q_table.loc[state, action])
        self.q_table.loc[state, action] = self.q_table.loc[state, action] + temp
    
# 一维环境
# 10个位置，1~10，终点在10
class Environment(object):
    
    def __init__(self, init_state=1, goal_state=10):
        self.state = init_state
        self.goal_state = goal_state
        self.action = 0
        self.map = list('----------')
        self.map[self.state-1] = 'o'
        self.map[self.goal_state-1] = '#'
    
    def reset(self, init_state=1, goal_state=10):
        self.state = init_state
        self.goal_state = goal_state
        self.action = 0
        self.map = list('----------')
        self.map[self.state-1] = 'o'
        self.map[self.goal_state-1] = '#'
        
    def step(self, action):
        self.map[self.state-1] = '-'
        self.state = self.state + action
        self.state = max(1, min(10, self.state))
        self.map[self.state-1] = 'o'
        
        if self.state == 10:
            reward = 10
            is_terminated = True
        else:
            reward = 0
            is_terminated = False
        return (self.state, reward, is_terminated)
        
    def show(self):
        print('\r'+''.join(self.map), end='')
        
        
if __name__ == '__main__':
    
    refresh_time = 0.1
    max_episode = 20
    
    env = Environment()
    agt = Agent((1,2,3,4,5,6,7,8,9,10), (-1,0,1))
    
    for i in range(max_episode):
        env.reset()
        is_terminated = False
        while True:
            env.show()
            time.sleep(refresh_time)
            if is_terminated:
                break
            s = env.state
            a = agt.choose_action(s)
            sn, r, is_terminated = env.step(a)
            agt.update(s, a, r, sn)
        print('')
        print(agt.q_table)

```

