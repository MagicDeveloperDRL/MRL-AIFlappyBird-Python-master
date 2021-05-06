'''''''''
@file: dqn.py
@author: MRL Liu
@time: 2021/2/21 20:15
@env: Python,Numpy
@desc: 基于DQN算法的AI大脑
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import os
import random

import  numpy as np
import tensorflow as tf # 使用tf来构建神经网络
from replaybuffer import ReplayBuffer



FRAME_PER_ACTION = 1 # 多少帧采取一次动作
MODEL_SAVE_PATH='./models/'
LOGS_SAVE_PATH='./logs/'
MODEL_NAME='model.ckpt'
OBSERVE = 100
class DQN(object):
    # 初始化参数
    def __init__(
            self,
            n_actions,  # 动作个数
            learning_rate=1e-6,  # 学习率
            e_greedy=0.8,  # e-greedy
            e_greedy_increment=0.0001,  # 是否让greedy变化

            batch_size=32,  # 每次采样数据的大小
            memory_size=50000,  # 记忆池的行数据大小
            gamma=0.99,  # 回报折扣因子
            replace_target_iter=300,
            output_graph=False,  # 是否输出TensorBoard
            save_model=True,
            read_saved_model=False,
    ):
        self.n_actions = n_actions
        # 贪婪值
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # 更新相关参数
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_rate = learning_rate
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        # 相关设置
        self.output_graph = output_graph
        self.save_model = save_model
        self.read_saved_model = read_saved_model
        # 保存路径
        self.model_save_path = MODEL_SAVE_PATH
        self.logs_save_path = LOGS_SAVE_PATH
        self.model_name = MODEL_NAME
        # 初始化经验池
        self.memory = ReplayBuffer(self.memory_size)
        # 初始化计算图
        self.define_graph(None)


    # 选择动作(epsilon greedy)
    def choose_action(self, state):
        QValue = self.q_eval.eval(feed_dict={self.s: [state]})[0] # 获取两个动作的价值
        action = np.zeros(self.n_actions)
        if np.random.uniform() > self.epsilon:
            action_index = np.random.randint(0,self.n_actions)# 获得随机的一个方向
            action[action_index] = 1
            #print('随机挑选:',action_index)
        else:
            action_index = np.argmax(QValue) # 获得当前位置两个方向较大的索引值
            action[action_index] = 1
            #print('AI挑选：',action_index)
        return action

    def update(self, s, a, r, s_,done):
        self.store_in_memory(s, a, r, s_,done)
        # 以固定频率进行学习本回合的经验(s, a, r, s)
        if (self.step_counter > self.memory_size) and (self.step_counter % 5 == 0):
            self.learn()
        elif self.step_counter% 20 == 0:  # 指定目录下打印消息
            print("已经收集{}条数据".format(self.memory.timeStep))
            #self.memory.save_memory_json()  # 保存数据
        self.step_counter+=1
    # 学习策略
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_net的参数被更新\n')
        # Step 1:  # 获取批次数据
        minibatch =self.memory.sample(self.batch_size) # 获得一个batch的图片信息
        state_batch = [data[0] for data in minibatch] # 获得状态信息, [80, 80, 4]
        action_batch = [np.argmax(data[1]) for data in minibatch]# 获得动作信息，即向上的索引值为[1, 0], 向下的为[0, 1]
        reward_batch = [data[2] for data in minibatch]# 获取奖励信息
        nextState_batch = [data[3] for data in minibatch]# 获得下一状态信息, [80, 80, 4]
        terminal_batch = [data[4] for data in minibatch]# 获得是否合格, [80, 80, 4]


        # 获取后继状态的后继Q值
        q_next_batch = self.q_next.eval(feed_dict={self.s: nextState_batch})
        # 根据Q-Learning机制计算目标Q值
        q_eval_batch = self.sess.run(self.q_eval, {self.s: state_batch})  # 使用评估网络获取当前状态的当前Q值
        q_target_batch = q_eval_batch.copy()  # 目标Q值和当前Q值具有相同的矩阵结构，所以直接复制
        for i in range(0, self.batch_size):
            terminal = terminal_batch[i]
            if terminal:
                q_target_batch[i, action_batch[i]] = reward_batch[i]  + self.gamma * np.max(q_next_batch[i])  # 目标Q值=当前奖励+折扣因子*后继Q值
            else:
                q_target_batch[i, action_batch[i]] = reward_batch[i]  # 目标Q值=当前奖励


        # 定时保存网络
        if self.save_model and self.learn_step_counter % 1000 == 0:
            self.saver.save(self.sess, os.path.join(self.model_save_path, self.model_name), global_step=self.learn_step_counter)
            _, cost = self.sess.run([self.trainStep, self.cost],
                                    feed_dict={self.s: state_batch,
                                               self.q_target: q_target_batch})
            print('learn_step: %d , loss: %g. and save model successfully' % (self.learn_step_counter, cost))
        elif self.output_graph and self.learn_step_counter % 10 == 0:
            _, cost,summary = self.sess.run([self.trainStep, self.cost,self.merged_summary_op],
                                    feed_dict={self.s: state_batch,
                                               self.q_target: q_target_batch})
            self.train_writer.add_summary(summary, self.learn_step_counter)  # 添加日志
            print('learn_step: %d , loss: %g, epsilon:%g and add logs successfully' % (self.learn_step_counter, cost,self.epsilon))
        elif self.learn_step_counter % 10 == 0:
            _, cost = self.sess.run([self.trainStep, self.cost],
                                    feed_dict={self.s: state_batch,
                                               self.q_target: q_target_batch})
            print('learn_step: %d , loss: %g, epsilon:%g' % (self.learn_step_counter, cost,self.epsilon))
        # 逐步提高的利用概率,让算法尽快收敛的编程技巧
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1



    # 向记忆池存入数据
    def store_in_memory(self, Observation,action,reward,nextObservation,terminal):
        self.memory.store_transition(Observation,action,reward,nextObservation,terminal)






    def define_graph(self,sess):
        # 定义目标网络的输入输出
        self.s = tf.placeholder(dtype="float", shape=[None, 80, 80, 4], name='s')
        self.q_target = tf.placeholder(dtype="float", shape=[None,self.n_actions], name='q_target')
        with tf.variable_scope('target_net'):
            with tf.variable_scope('output'):
                self.q_next = self._define_cnn_net(self.s, c_names=['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES])
        # 定义评估网络的输入输出
        with tf.variable_scope('eval_net'):
            with tf.variable_scope('output'):
                self.q_eval = self._define_cnn_net(self.s,c_names=['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES])
            with tf.variable_scope('loss'):
                #Q_action = tf.reduce_sum(tf.multiply(self.q_eval, self.a), reduction_indices=1)# 使用位置信息与预测的奖励值进行相乘操作, 获得当前位置的预测奖励值
                self.cost = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))# 使用预测奖励值与当前位置的奖励平方差来获得损失值
                tf.summary.scalar("loss", self.cost)  # 使用TensorBoard监测该变量
            with tf.variable_scope('train'):
                self.trainStep = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)
        # 定义变量
        self.learn_step_counter = 0
        self.step_counter = 0
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.merged_summary_op = tf.summary.merge_all()  # 合并所有的summary为一个操作节点，方便运行
        # 初始化会话：
        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess
        # 是否保存模型：
        if self.save_model:
            self.saver = tf.train.Saver()  # 网络模型保存器
        else:
            self.saver = None
        # 是否导入模型：
        if self.read_saved_model:
            checkpoint = tf.train.get_checkpoint_state(os.path.join(self.model_save_path,""))
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                strNum = checkpoint.model_checkpoint_path.split(',')[-1].split('-')[-1]
                self.learn_step_counter = int(strNum)
                print("成功读取指定模型：", checkpoint.model_checkpoint_path)
            else:
                print("无法找到指定的checkpoint文件")
        # 是否输出图
        if self.output_graph:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.logs_save_path, ""), self.sess.graph)
        else:
            self.train_writer = None
        self.sess.run(tf.global_variables_initializer())

    def _define_cnn_net(self,input,c_names):
        # 定义卷积池化层L1
        layer_conv1 = self._define_conv2d_layer(inputs=input,
                                                conv_filter_size=8,
                                                num_input_channels=4,
                                                num_filters=32,
                                                stride =4 ,
                                                activation_function=tf.nn.relu,
                                                c_names = c_names,
                                                layer_name = 'layer_conv1')
        print('layer_conv1',layer_conv1.get_shape())
        layer_conv_pool1 = self.__maxpool2d(layer_conv1)
        print('layer_conv_pool1', layer_conv_pool1.get_shape())
        # 定义卷积层L2
        layer_conv2 = self._define_conv2d_layer(inputs=layer_conv_pool1,
                                                conv_filter_size=4,
                                                num_input_channels=32,
                                                num_filters=64,
                                                stride=2,
                                                activation_function=tf.nn.relu,
                                                c_names=c_names,
                                                layer_name='layer_conv2')
        print('layer_conv2', layer_conv2.get_shape())
        # 定义卷积层L3
        layer_conv3 = self._define_conv2d_layer(inputs=layer_conv2,
                                                conv_filter_size=3,
                                                num_input_channels=64,
                                                num_filters=64,
                                                stride=1,
                                                activation_function=tf.nn.relu,
                                                c_names=c_names,
                                                layer_name='layer_conv3')
        #layer_conv3_flat = tf.reshape(layer_conv3, [-1, 1600])
        print('layer_conv3', layer_conv3.get_shape())
        layer_conv3_flat = self.__flattenlayer(layer_conv3)
        print('layer_conv3_flat', layer_conv3_flat.get_shape())
        # 定义全连接层L4
        layer_fnn4 = self._define_fc_layer(inputs=layer_conv3_flat,
                                                num_inputs=1600,
                                                num_outputs=512,
                                                activation_function=tf.nn.relu,
                                                c_names=c_names,
                                                layer_name='layer_fnn4')
        print('layer_fnn4', layer_fnn4.get_shape())
        # 定义全连接层L5
        output = self._define_fc_layer(inputs=layer_fnn4,
                                       num_inputs=512,
                                       num_outputs=self.n_actions,
                                       activation_function=None,
                                       c_names=c_names,
                                       layer_name='layer_fnn5')
        print('output', output.get_shape())
        return output

    def _define_fc_layer(self,inputs, # 输入数据
                         num_inputs,# 输入通道数
                         num_outputs,# 输出通道数
                         activation_function, # 激活函数
                         layer_name,  # 卷积层名字
                         c_names=None,
                         regularizer__function=None,
                         is_historgram=True):
        """ 定义一个全连接神经层"""
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            weights = self.__define_weights(shape=[num_inputs, num_outputs], c_names=c_names, regularizer__function=regularizer__function)
            biases = self.__define_biases(size=num_outputs, c_names=c_names)
            with tf.variable_scope('wx_plus_b'):
                # 神经元未激活的值，矩阵乘法size
                wx_plus_b = tf.matmul(inputs, weights) + biases
            # 使用激活函数进行激活
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                tf.summary.histogram(layer_name + '/outputs', outputs)
                # 返回神经层的输出
        return outputs



    def _define_conv2d_layer(self,inputs,# 输入数据
                             num_input_channels,# 输入通道数
                             conv_filter_size, # 卷积核尺寸
                             num_filters,# 卷积核数量，即输出通道数
                             stride ,# 卷积核步长
                             activation_function, # 激活函数
                             layer_name, # 卷积层名字
                             c_names=None,
                             regularizer__function=None,
                             is_historgram=True):
        """ 定义一个卷积神经层"""
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            weights = self.__define_weights( shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters],c_names=c_names,regularizer__function=regularizer__function)
            biases = self.__define_biases( size=num_filters,c_names=c_names)
            with tf.variable_scope('conv_plus_b'):
                # 神经元未激活的值，卷积操作
                conv_plus_b = self.__conv2d(inputs, weights, stride) + biases
            # 使用激活函数进行激活
            if activation_function is None:
                outputs = conv_plus_b
            else:
                outputs = activation_function(conv_plus_b)
            if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                tf.summary.histogram(layer_name + '/outputs', outputs)
        # 返回神经层的输出
        return outputs

    def __define_weights(self, shape,c_names=None,regularizer__function=None):
        with tf.variable_scope('weights'):
            weights = tf.Variable(initial_value=tf.truncated_normal(shape, stddev=0.01),name='w',  collections=c_names)
            if regularizer__function != None:  # 是否使用正则化项
                tf.add_to_collection('losses', regularizer__function(weights))  # 将正则项添加到一个名为'losses'的列表中
        return weights

    def __define_biases(self, size,c_names=None):
        with tf.variable_scope('biases'):
            biases = tf.Variable(initial_value=tf.constant(0.01, shape=[size]),name='b',collections=c_names)
        return biases

    def __conv2d(self, input, weights, stride, padding='SAME'):
        layer = tf.nn.conv2d(input=input,  # 输入的原始张量
                             filter=weights,  # 卷积核张量，(filter_height、 filter_width、in_channels,out_channels)
                             strides=[1, stride, stride, 1],
                             padding=padding)
        return layer

    def __maxpool2d(self, input, stride=2, padding='SAME'):
        layer = tf.nn.max_pool(value=input,  # 这是一个float32元素和形状的四维张量（批长度、高度、宽度和通道）
                               ksize=[1, stride, stride, 1],  # 一个整型list，表示每个维度的窗口大小
                               strides=[1, stride, stride, 1],  # 在每个维度上移动窗口的步长。
                               padding=padding)  # VALID或SAME
        return layer

    def __flattenlayer(self, layer):
        layer_shape = layer.get_shape()  # 扁平前 (?, 8, 8, 64)
        num_features = layer_shape[1:4].num_elements()  # [1:4]: (8, 8, 64),num_features: 4096
        re_layer = tf.reshape(layer, [-1, num_features])  # 扁平后 (?, 4096)
        return re_layer

