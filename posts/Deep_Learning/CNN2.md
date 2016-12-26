## 概述

首先卷积神经网络(CNNs)是由一层或多层卷积层然后再连接一个或多层全连接层组成。通常情况下，卷积层后会跟着一个池化层/下采样层。当然也有很多人将卷积操作与池化操作组合起来看做是一个卷积层，如下图。

![layer_terminology](https://raw.githubusercontent.com/lehyu/lehyu.cn/master/image/DL/CNNs/layer_terminology.png)

CNNs能够很好的处理二维数据(比如说图片)，我们[前面](http://blog.csdn.net/lehyu/article/details/52266090)已经说过了，使用卷积层而不是全连接层能够使训练更加容易而且整个网络的参数也会减少。

## 卷积层的正向传播

假设当前层为卷积层，前一层输入的数据为 $M \times N$ 。使用一个 $m \times n$ 的核 $w$ ，来进行卷积操作，那么卷积层的输出为 $(M - m + 1) \times (N - n + 1)$ 。那么

$$\begin{equation}
\begin{array}{rcl}
z_{ij}^{l} = \sum_{a = 0}^{m-1} \sum_{b=0}^{n-1} w_{ab} y_{(i+a)(j+b)}^{l-1} + b
\end{array}
\end{equation}$$

更形象的操作如下图

![Convolutions](https://raw.githubusercontent.com/lehyu/lehyu.cn/master/image/DL/CNNs/Convolution_schematic.gif)

再经非线性变换后得到：$y_{ij}^{l} = f_{l-1}(z_{ij}^{l})$ 。这里的非线性变换就是detector stage。

## 池化层的正向传播

假设池化层的输入为 $M \times N$，将 $m \times n$ 看做是一个池，其中 $m \leqq M \land M \% m = 0$，同样对于 $N,n$ 也有同样的要求。那么输出为 $\frac{M}{m} \times \frac{N}{n}$。

池化操作有很多类型，比如Max-pooling, mean-pooling。 通常情况下，可以将池化操作看做是非重叠的卷积操作，这样可以调用第三方的库进行加速优化，减少工作量。
举个例子，Max-pooling：

$$\begin{equation}
\begin{array}{rcl}
z_{ij}^{l}= \max\{y_{(i+0)(j+0)}^{l-1},y_{(i+1)(j+0)}^{l-1},\dots,y_{(i+m-1)(j+n-1)}^{l-1}\}
\end{array}
\end{equation}$$

有些应用，可能会在之后再添加一个非线性变换，这种情况比较少见。

## 池化层的反向传播

池化层的下一层可能是全连接层也可能是卷积层，对于全连接层来说，与[之前](http://blog.csdn.net/lehyu/article/details/52232063)讨论的一样。假设当前池化层为第 $l$ 层，则它的 error term为 $\delta^{l}$:

$$\begin{equation}
\begin{array}{rcl}
\delta^{l} = \delta^{l+1} W^{l+1}
\end{array}
\end{equation}$$

//todo 下一层卷积层时的敏感误差项

## 卷积层的反向传播
卷积层的下一层一般为池化层，卷积层到池化层其实只做了一个池化操作。假设卷积层一共有 $k$ 个 feature maps。卷积层每一个feature maps做一个池化操作，得到 $k$ 个池，也就是说卷积层与池化层存在一个一一对应的关系。

如果想求第 $j$ 个feature map的误差敏感项 $\delta_{j}^{l}$:

$$\begin{equation}
\begin{array}{rcl}
\delta_{j}^{l} = upsample(\delta_{j}^{l+1})
\end{array}
\end{equation}$$

上面这个是池化操作后没有经过非线性变换的误差敏感项的求法，但是如果添加一个非线性变换的话，可能会更复杂一些，可以参考[Jake Bouvrie的笔记](http://cogprints.org/5869/1/cnn_tutorial.pdf)。
