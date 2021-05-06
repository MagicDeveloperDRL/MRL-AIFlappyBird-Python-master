# MRL-AIFlappyBird-Python-master
# 使用AI算法玩FlappyBird

## ——基于DQN的RL算法 

本项目是用Python编写的AI小游戏，是作者实现用AI玩Flappy Bird小游戏的小项目，当然为了便于AI训练，该游戏进行了一定的简化处理，没有添加开始游戏等其他界面。本文使用的AI算法是基于TensorFlow框架的Nature DQN算法。

# 一、Flappy Bird是什么

Flappy Bird是一款玩家要在游戏中尽可能长地维持小鸟生命的游戏。

小鸟不断向前飞行，会遇到一系列高低不同的管道，管道将小鸟通过的高度限制在特定的范围内。

小鸟由于重力会自动掉落到地面，所以玩家需要不断操作使小鸟进行Flap，躲避管道和地面，游戏分数由小鸟成功通过多少个管道障碍物来衡量。

如果小鸟撞到地面或者管道，它就会死亡并结束游戏。

# 二、游戏实现效果

![20210502_212711.gif](https://img-blog.csdnimg.cn/img_convert/724c865c68227b7a9afe0b9a585bf78d.gif)

# 三、计算图结构

![img](https://img-blog.csdnimg.cn/2021050415403249.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxOTU5OTIw,size_16,color_FFFFFF,t_70)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

该项目的相关博客链接：

【Python小游戏】用AI玩Python小游戏FlappyBird【源码】（https://blog.csdn.net/qq_41959920/article/details/116357264?spm=1001.2014.3001.5501 ）