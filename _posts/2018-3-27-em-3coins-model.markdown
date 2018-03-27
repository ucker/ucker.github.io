---
layout: post
title:  "EM Algorithm for 3 Coins Model"
date:   2018-03-27
categories: blog
---
# 三硬币模型EM算法

>模型来自《统计学习方法》，在看了brml后，我用brml中描述的EM算法解决三硬币模型

## 模型描述
假设有3枚硬币，分别记作A、B、C，这些硬币正面出现的概率分别是$\pi$，p和q。进行如下掷硬币试验：先掷硬币A，根据其结果选出硬币B或硬币C，正面选硬币B，反面选硬币C；然后掷选出的硬币，掷硬币的结果，出现正面记作1，出现反面记作0；独立地重复N次实验（这里，$N=10$），观察地结果如下：

$$
1,1,0,1,0,0,1,0,1,1
$$
假设只能观测到掷硬币地结果，不能观测掷硬币的过程，问如何估计三硬币正面出现的概率，即三硬币模型的参数。

## E-step
设$v$是观测到的随机变量，$h$是隐藏随机变量，$\theta$是需要学习的参数。这里$v$表示观测的硬币的结果，$dom(v)=\{0,1\}$，$h$表示观测结果来自哪个硬币，1表示来自B硬币，0表示来自C硬币，$dom(h)=\{0,1\}$。

已知

$$
<\log{q(h|v)}>_{q(h|v)}-<\log{p(h,v|\theta)}>_{q(h|v)}+\log{p(v|\theta)}=KL(q(h|v)|p(h|v,\theta)) \geq 0
$$
最优化使KL为0，则要使得

$$
q(h|v)=p(h|v,\theta)
$$
记$q(h=x|v=y)=q_{x,y}$，在三硬币模型中E-step的更新公式如下

$$
q^{new}_{1,1}=\frac{\pi p}{\pi p + (1-\pi)q}
$$
$$
q^{new}_{0,1}=\frac{(1-\pi)q}{\pi p + (1-\pi)q}
$$
$$
q^{new}_{1,0}=\frac{\pi (1-p)}{\pi (1-p) + (1-\pi)(1-q)}
$$
$$
q^{new}_{0,0}=\frac{(1-\pi)(1-q)}{\pi (1-p) + (1-\pi)(1-q)}
$$

## M-step
从brml的观点来看，就是最大化$E(\theta)=<\log{p(h,v|\theta)}>_{q(h|v)}$(brml把这一项称为Energy)。把这个期望展开

$$
E(\theta)=\#(v=1)(q_{1,1}\log{p \pi}+q_{0,1}\log{q(1-\pi)})+\#(v=0)(q_{1,0}\log{(1-p)\pi}+q_{0,0}\log{(1-q)(1-\pi)})
$$
分别对$\pi,p,q$求导可得M-step的更新公式

$$
\pi^{new}=\frac{\#(v=1)q_{1,1}+\#(v=0)q_{1,0}}{N}
$$
$$
p^{new}=\frac{\#(v=1)q_{1,1}}{\#(v=1)q_{1,1}+\#(v=0)q_{1,0}}
$$
$$
q^{new}=\frac{\#(v=1)q_{0,1}}{\#(v=1)q_{0,1}+\#(v=0)q_{0,0}}
$$

## 程序

设置和书中例子一样的初始值

```python
import numpy as np
data = np.array([1,1,0,1,0,0,1,0,1,1])
head = (data == 1).sum()
tail = (data == 0).sum()
def em(pi=0.5, p=0.5, q=0.5):
    e_step = np.zeros((4,1))
    for i in range(10000):
        e_step[0] = pi * p / (pi * p + (1 - pi)*q)
        e_step[1] = (1 - pi) * q / (pi * p + (1 - pi)*q)
        e_step[2] = pi * (1 - p) / (pi * (1 - p) + (1-pi)* (1 - q))
        e_step[3] = (1-pi)* (1 - q) / (pi * (1 - p) + (1-pi)* (1 - q))
        pi = (head * e_step[0] + tail * e_step[2]) / (head + tail)
        p = head * e_step[0] / (head * e_step[0] + tail * e_step[2])
        q = head * e_step[1] / (head * e_step[1] + tail * e_step[3])
        return (pi, p, q)

print(em())
print(em(0.4, 0.6, 0.7))
```
结果
```bash
> python em.py
(array([0.5]), array([0.6]), array([0.6]))
(array([0.40641711]), array([0.53684211]), array([0.64324324]))
```
