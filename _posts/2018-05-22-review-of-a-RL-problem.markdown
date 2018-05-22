---
layout: post
title:  "Review of A Reinforcement Learning Problem"
date:   2018-05-22
categories: blog
---
# 一个强化学习问题的回顾
  
  >本文仅仅是为了把我最近学到的东西应用起来，其中省略了很多的背景知识的介绍
  
  问题来自[http://mnemstudio.org/path-finding-q-learning-tutorial.htm](http://mnemstudio.org/path-finding-q-learning-tutorial.htm)。原文中是使用Q-Learning解决问题的，最近看了些贝叶斯网络拓展的内容（Extending Bayesian Networks for Decision），这部分和概率图一脉相承，主要引出了Influence Diagrams的概念，在贝叶斯网络中加入决定节点（Decision Nodes）和价值节点（Utility Nodes），最优化目标一般是Expected Utility。只需要根据ID（Influence Diagrams）中节点的偏序关系就可以表示出图的Expected Utility。
  
  在这个问题中只需要用到ID的一种简单形式——MDP（Markov Decision Process）。因为使用MDP问题描述起来会简单很多。下面介绍解决问题需要用到的核心内容。

## Bellman's Equation
  
  Bellman's Equation有如下的形式

  $$
  v_{t-1}(x_{t-1})=u(x_{t-1})+\underset{d_{t-1}}{\mathrm{max}}\sum_{x_{t}}p(x_{t}|x_{t-1},d_{t-1})v_{t-1}(x_{t})
  $$

  其中$v_{t-1}(x_{t-1})$是为了做Message Passing时方便而额外定义的一个标记，使用这个标记在求解时会方便很多。其中$u(x_{t-1})$时在$x_{t-1}$状态的Utility。其他的符号应该不需要解释了。

  MDP问题的最终目的为了在某种状态下做出决策，于是有

  $$
  d_{t}^{*}=\underset{d_{t}}{\mathrm{argmax}}\sum_{x_{t+1}}p(x_{t+1}|x_{t},d_{t})v(x_{t+1})
  $$

## Solution to this Problem  
  
  这个问题可以看作是一个无穷状态的问题（虽然有终止状态，但是并不知道什么时候停止），因此需要将每个$u(x_{t})$改为$\gamma\ u(x_{t})(\gamma \in (0,1)$（这样Utility之和的上界就是一个等比数列，因此极限存在）。这时相应的Bellman's Equation变为

  $$
  v(s)=u(s)+\gamma \underset{d}{\mathrm{max}}\sum_{s'}p(x_{t}=s'|x_{t-1}=s,d_{t-1}=d)v(s')
  $$

  做决定使用

  $$
  d^{*}(s)=\underset{d}{\mathrm{argmax}}\sum_{s'}p(x_{t+1}=s'|x_{t}=s,d_{t}=d)v(s')
  $$
  
  对问题的进一步分析可以知道$p(x_{t}=s'|x_{t-1}=s,d_{t-1}=d)$只能取0或1，因此$max_{d}\sum_{s'}p(x_{t}=s'|x_{t-1}=s,d_{t-1}=d)v(s')$等于由状态$s$可以到达的状态$s'$中的最大的$v(s')$。我们把转移图的邻接矩阵记为$T$，Bellman's Equation变为

  $$
  v(s)=u(s)+\gamma \underset{s'}{\mathrm{max}}\{T(s,s')v(s')\}
  $$

  做决策（写的不准确，$s$与$d$是有区别的，这个问题很特殊，为了方便不做区分）

  $$
  d^{*}(s)=\underset{s'}{\mathrm{argmax}}\{T(s,s')v(s')\}
  $$
  
  这个问题中$p(s'|s,d)$是已知的，要求解的是$v$（也可以看作是另外一种$u$，具体和Message Passing有关）。采用Value Iteration的方式求和，也就是对$v$不断迭代，直至收敛，下面是实现的细节。

## Algorithm and Code
  
  先给出图的邻接矩阵，然后初始化$v,u$，赋值都是$\{0,0,0,0,0,100\}$。给定初始状态，根据Bellman's Equation更新状态和修改$v$的值，如果状态变为5（最终状态），重设初始状态，继续迭代。代码实现中的一些细节会以注释的形式给出。
```python
import numpy as np
from random import randint
v = np.array([0,0,0,0,0,100])
u = np.array(v)  # 0 1 2 3 4 5
state = np.array([[0,0,0,0,1,0],  # 0
                  [0,0,0,1,0,1],  # 1
                  [0,0,0,1,0,0],  # 2
                  [0,1,1,0,1,0],  # 3
                  [1,0,0,1,0,1],  # 4
                  [0,1,0,0,1,1]]) # 5
s = 2
iter_num = 10000
end = 0
gamma = 0.9
for i in range(iter_num):
    #print(s)
    neighbour = state[s,:]
    value = []
    for i in range(len(neighbour)):
        if neighbour[i] == 0:
            value.append(-1)
        else:
            value.append(v[i])
    max_val = np.max(value)
    arg_max_list = []
    for i in range(len(value)):
        if value[i] == max_val:
            arg_max_list.append(i)
    # 从多个最优状态中随机取出一个
    arg_max = arg_max_list[randint(0, len(arg_max_list)-1)]
    v[s] = u[s] + gamma * max_val
    # 如果到达最终状态，就随机选择一个非最终状态
    # 否则更新状态
    if s == arg_max and s == 5:
        s = randint(0, 4)
    else:
        s = arg_max
print(v)
```
  
  结果是
```bash
[801 891 720 801 891 991]
```
## Comparison
  
  相比Q-Learning我觉得这种方法建模更加的清楚（我认为这也是几乎所有概率图模型的优点）。值得一提的是这种方法对$\gamma$的解释更加的本质一些，在Q-Learning中$\gamma$又被称为学习率。

## Notes
  
  上面的介绍省去了很多内容，想弄清楚可以进一步阅读brml chapter 7。当然想要完全理解这些内容，必须要知道概率图的知识（Bellman's Equation的推导是Inference中常用的技巧）。
