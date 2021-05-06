'''''''''
@file: FlappyBirdTrainer.py
@author: MRL Liu
@time: 2021/2/21 20:10
@env: Python,Numpy
@desc: 训练器
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import sys
import cv2
import numpy as np
sys.path.append("game/")
import Flappy_Bird_Env as game
import dqn as agent

OBSERVE = 100

class FlappyBird_Trainer(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self,max_episodes):
        print('\n仿真训练启动...')
        step_counter = 0
        for episode in range(max_episodes):
            # 获取初始环境状态
            action0 = np.array([1, 0])  # 初始动作为向下降落
            x_rgb, reward, done = self.env.frame_step(action0)
            state = self.get_init_state(x_rgb)  # 对后继状态进行预处理
            # 开始本回合的仿真
            while True:
                action = self.agent.choose_action(state)  # agent根据当前状态采取动作
                #action = np.zeros(2)
                #action[1]=1
                #if action[0]==1:
                    #print('小鸟下降')
                #if action[1]==1:
                    #print('小鸟上升')
                x_rgb, reward, done = self.env.frame_step(action)  # env根据动作做出反馈
                # 将转换元组存入记忆池
                state_ = self.get_next_state(state,x_rgb) # 对后继状态进行预处理
                self.agent.store_in_memory(state,action,reward,state_,done)
                # 学习本回合的经验(s, a, r, s)
                if step_counter>OBSERVE:
                    self.agent.learn()
                # 当前状态发生切换
                state = state_
                step_counter += 1
        print('\n仿真训练结束')

    # 预处理函数，获取初始状态
    def get_init_state(self, x_rgb):
        x_rgb = cv2.resize(x_rgb, (80, 80))  # 将RGB图像进行缩放
        x_gray = cv2.cvtColor(x_rgb, cv2.COLOR_BGR2GRAY)  # 将RGB图像转换为灰度图像
        ret, x_t = cv2.threshold(x_gray, 1, 255, cv2.THRESH_BINARY)  # 将灰度图像进行二值处理，将大于等于1的索引变化为255，等于0的不变
        state = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 将一帧图像拼接，使得图片的维度为[80, 80, 4]
        return state
    # 预处理函数，将当前状态和新一帧画面处理成新状态
    def get_next_state(self, state, x_rgb):
        x_rgb = cv2.resize(x_rgb, (80, 80))  # 将RGB图像进行缩放
        x_gray = cv2.cvtColor(x_rgb, cv2.COLOR_BGR2GRAY)  # 将RGB图像转换为灰度图像
        ret, x_t = cv2.threshold(x_gray, 1, 255, cv2.THRESH_BINARY)  # 将灰度图像进行二值处理，将大于等于1的索引变化为255，等于0的不变
        x_t = np.reshape(x_t, (80, 80, 1))  # 将维度转换为(80, 80, 1)
        next_state = np.append(x_t, state[:, :, :3], axis=2)  # 与上一个图片状态的前3帧图片进行串接操作
        return next_state

if __name__=='__main__':
    # 初始化env
    env = game.Flappy_Bird_Env()
    # 初始化agent
    agent = agent.DQN(n_actions=2, # 动作空间个数
                      output_graph = False,# 是否输出日志
                      save_model = True, # 是否保存训练中的模型
                      read_saved_model = True, # 是否读取已有的模型
                      e_greedy_increment=0.0001,# 是否让greedy变化,设置为None，标志着agent的探索能力
                      )
    # 初始化训练器
    trainer = FlappyBird_Trainer(env=env,agent=agent)
    # 开始训练
    trainer.train(max_episodes=1)


