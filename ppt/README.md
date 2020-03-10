# STGCN：Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting

the paper "[Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/pdf/1709.04875.pdf)"

## 问题定义

如何准确的进行中长期的交通预测（中长期：over 30 minutes）

本篇论文主要是对地点的速度进行预测

## 前人工作

在这篇论文之前也有几种交通预测的方法：动态建模，数据驱动到后来的深度学习的一些方法。但是传统的方法都是将数据看作是网格数据，而且对于中长期的预测效果也并不好。

![Previous_work1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/Previous_work1.jpeg)

![Previous_work2](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/Previous_work2.jpeg)

以下两个图来自两篇论文，具体请见参考文献[1]，[2]。

![Previous_work2](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/Previous_work3.png)

## 本篇论文提出的解决方法

首先将网格数据改为图数据作为输入，图我们都知道可以用邻接矩阵来表示，就像图中的W就是图的邻接矩阵，实验中使用的数据集PeMSD7(M)共有228个数据点，相当于一个具有228个顶点的图，因为这个模型主要是对速度进行预测，所以每个顶点只有一个特征就是：速度。

之后提出了一个结构ST-Conv Block来对时空进行建模。

![Previous_work1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/method.jpeg)

## 模型架构

图卷积的相关内容请见：[动态理解图卷积](https://github.com/Knowledge-Precipitation-Tribe/Graph-neural-network#动态理解图卷积)

<div align = "center"><image src="https://github.com/Knowledge-Precipitation-Tribe/Graph-neural-network/blob/master/images/GCN4.gif" width = "300" height = "240" alt="axis" align=center /></div>
动态理解图卷积正在完善...

### Graph CNNs for Extracting Spatial Features

首先使用图卷积来捕获空间相关性，本篇论文采用的是切比雪夫近似与一阶近似后的图卷积公式，我们只看最终的那个卷积公式，其中D为图的度矩阵，A_hat为图的邻接矩阵+单位矩阵，为的是在卷积过程中不仅考虑邻居节点的状态，也考虑自身的状态。

![model1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model1.jpeg)

### Gated CNNs for Extracting Temporal Features

在时间维度上采用门控卷积来捕获时间依赖性，而且与传统的卷积方法不同，因为这要考虑时间序列的问题，所以这里采用的是因果卷积。因为我们使用卷积操作，就不用像以前的采用RNN的方法依赖于之前的输出，所以我们可以对数据进行并行计算，这样使得模型训练速度更快。

而且采用还采用了GLU操作，GLU是在这篇论文中提出的：[Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf)。在STGCN这篇论文中作者并没有对此进行过多的解释，我的理解是采用这种操作可以缓解梯度消失等现象还可以保留模型的非线性能力。

![model2](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model2.jpeg)

而且我们可以看到模型的运行效果也与论文中的描述一致，当考虑时间维度上的Kt个邻居时输出序列的长度就会减少Kt-1。在代码中Kt为3，输入的时间维度是12，卷积之后的数据结果就为10。

![model3](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model3.png)

### ST-Conv Block

将以上的图卷积和门控CNN组合成如图所示的结构，其中使用了瓶颈策略来实现尺度压缩和特征压缩。并在每层之后接归一化层来防止过拟合。

ST-Conv Block的公式就是图的另一个解释，输入数据先做时间维度卷积，输出结果再做图卷积，图卷积的输出结果经过一个RELU，在进行一个时间维度卷积，就是整个ST-Conv Block的输出。

![model4](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/model4.jpeg)

### Model

最后的模型是堆叠两个St-Conv Block之后接一个输出层，其中输出层首先用时间维度的卷积将之前的输出数据的时间维度进行合并，合并之后在经过一个卷积输出最终的预测数据，预测数据就是下一个时间维度的一张图[1,228,1]。模型采用的是L2损失。

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
图示为PeMSD7首页展示的路网。

对于PeMSD7分为两个数据：PeMSD7(M)和PeMSD7(L)。因为是每五分钟汇总一次数据，故一小时会产生12条数据，所以一天会产生288条数据

![experiment1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/PeMSD72.png)

在PeMSD7上，可以看到STGCN与其他几个模型相比效果还是很好的。

![PeMSD72](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/experiment1.png)

STGCN比其他方法更准确地捕捉到了高峰时间的趋势，而且它比其他模型更早察觉到高峰时间的结束。

![experiment1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/experiment2.png)

而且作者也用STGCN与GCGRU进行了对比，可以看到STGCN具有很高的计算效率。

![experiment1](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/experiment3.png)

在Titan XP双卡i7-8700K，12核32G的配置下，bathc_size为50的情况下，一个epoch训练时间大概在8秒左右。

![learning_rate](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/train.png)

在tensorboard中我们也可一看到在训练过程中的学习率与train_loss的变化。

![learning_rate](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/learning_rate.png)

![tarin_loss](https://github.com/Knowledge-Precipitation-Tribe/STGCN-keras/blob/master/ppt/images/train_loss.png)

## 参考文献

[1] [Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/pdf/1610.00081.pdf)

[2] [UrbanFM: Inferring Fine-Grained Urban Flows](https://arxiv.org/pdf/1902.05377.pdf)