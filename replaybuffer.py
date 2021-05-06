'''''''''
@file: ReplayBuffer.py
@author: MRL Liu
@time: 2021/4/20 17:08
@env: Python,Numpy
@desc: 基于队列的经验重放池
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
from collections import deque
import random
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory_size = capacity # 容量大小
        self.num = 0 # 存放的经验数据数量
        self.data = deque() # 存放经验数据的队列

    def store_transition(self, state,action,reward,state_,terminal):
        self.data.append((state, action, reward, state_, terminal))# 添加数据
        if len(self.data) > self.memory_size:
            self.data.popleft()
            self.num -= 1
        self.num += 1

    def sample(self, batch_size):
        minibatch = random.sample(self.data, batch_size)
        return minibatch  # 获取n个采样
