# STGCN：Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting

the paper "[Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/pdf/1709.04875.pdf)"

## 问题定义

如何准确的进行中长期的交通预测（中长期：over 30 minutes）

本篇论文主要是对地点的速度进行预测

## 前人工作

![Previous_work1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/Previous_work1.jpeg)

![Previous_work2](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/Previous_work2.jpeg)

![Previous_work2](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/Previous_work3.png)

![Previous_work2](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/Previous_work4.png)

下面两个图片来自参考文献[1]，[2]。

## 本篇论文提出的解决方法

![Previous_work1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/method.jpeg)

## 模型架构

[动态理解图卷积](https://github.com/Knowledge-Precipitation-Tribe/Graph-neural-network#动态理解图卷积)

<div align = "center"><image src="https://github.com/Knowledge-Precipitation-Tribe/Graph-neural-network/blob/master/images/GCN4.gif" width = "300" height = "240" alt="axis" align=center /></div>

![model1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model1.jpeg)

![model2](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model2.jpeg)

![model3](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model3.png)

![model4](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model4.jpeg)

![modelk5](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model5.jpeg)

![model6](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model6.png)

### 模型总结

- STGCN 是处理结构化时间序列的通用框架。它不仅能够解决交通网络建模和 预测问题，而且可以应用于更一般的时空序列学习任务。
- 时空卷积块结合了图卷积和门控时间卷积，能够提取出最有用的空间特征，并 连贯地捕捉到最基本的时间特征。
- 该模型完全由卷积结构组成，在输入端实现并行化，参数更少，训练速度更 快。更重要的是，这种经济架构允许模型以更高的效率处理大规模网络。

模型的总体结构我们可以通过tensorboard来查看

![model](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model.png)

## 实验

数据描述：北京市交通委员会和加利福尼亚交通运输署收集的两个现实世界交通数据集BJER4和PeMSD7

Dataset Description：two real-world traffic datasets, BJER4 and PeMSD7, collected by Beijing Municipal Traffic Commission and California Deportment of Transportation, respectively

PeMSD7网址：http://pems.dot.ca.gov/?dnode=Clearinghouse

<div align = "center"><image src="https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/PeMSD71.png" width = "300" height = "240" alt="axis" align=center /></div>

![experiment1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/PeMSD72.png)

![PeMSD72](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/experiment1.png)

![experiment1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/experiment2.png)

![experiment1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/experiment3.png)

![learning_rate](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/learning_rate.png)

![tarin_loss](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/train_loss.png)

## 参考文献

[1] [Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/pdf/1610.00081.pdf)

[2] [UrbanFM: Inferring Fine-Grained Urban Flows](https://arxiv.org/pdf/1902.05377.pdf)