前面几篇文章大致讲了卷积神经网络的基本构成，其中最难的就是卷积层的方向推导过程，但是这应该不是问题，因为现在很多架构都已经实现好了，只需要我们去配置模型就可以，当然，如果能够自己理解然后推导出来更加好。

下面介绍几个卷积神经网络的模型：AlexNet,GoogLeNet,VGG

### AlexNet

AlexNet是一个具有5层卷积层和3层全连接层的神经网络模型。这个模型的提出推动了深度学习的发展进程。AlexNet的结构如图所示：

![AlexNet](https://raw.githubusercontent.com/lehyu/lehyu.cn/master/image/DL/CNNs/AlexNet.png)

从上图我们可以知道，这个模型的训练分布在两块GPU上，只有在第三层卷积层和全连接层中才会有GPU之间的数据传递。

### 第一层卷积层

首先输入为 $227 \times 227 \times 3$ 而不是 $224$，由图可知道一共有 *96* 个大小为 $11 \times 11 \times 3$ 的卷积核，并且步幅为 *4* 。那么经卷积操作后，就得到 $55 \times 55 \times 96$ 的卷积层，其中 $(227-11)/4+1=55$，只就是为什么我们说输入为 *227* 的原因。这里需要注意的是，在卷积操作后，还要进行非线性变换，而AlexNet采用了[ReLu](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf)。

### 第二层卷积层

首先第一层卷积层经过ReLu后依然输出 $55 \times 55 \times 96$，但在输入到第二层卷积层之前还要经过大小为 $3 \times 3$ 而步幅为*2*的Max-pooling，输出为 $27 \times 27 \times 96$。第二层卷积层的核为 256个 $5 \times 5 \times 48$，跟[LetNet-5](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791)不一样，AlexNet将上一层的feature map都全连接到一块进行卷积操作。照理说应该输出为 $23 \times 23 \times 256$，但是AlexNet中给每个输入在外围添加了一个大小为 *2* 的pad，所以原输入为 $31 \times \times 31 \times 48$，那么输出就为 $27 \times 27 \times 256$。

同理在经过非线性变换。

### 第三层卷积层

第二层经Max-pooling后输出为 $13 \times 13 \times 256$。需要注意的是在第三层卷积层中，一共有 *384* 个 $3 \times 3 \times 256$ 的卷积核，这说明在第三层卷积层两块GPU之间有数据交互。而跟第二层有点类型，在输入的feature map外添加一个大小为 *1* 的pad，因此输出为 $13 \times 13 \times 384$。

同理在经过非线性变换。

### 第四层卷积层

需要注意的是在第三层与第四层卷积中间没用经过池化操作。其他的差不多。

具体细节可以参考caffe中的AlexNet的[配置文件](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt)

### Local Response Normalization

LRN操作在ReLu后执行，有利于泛化。假设 $a_{x,y}^{i}$ 是一个神经元在 $(x,y)$ 经kernel i计算而得的结果再通过ReLU变换而得，那么执行LRN操作后为 $b_{x,y}^{i}$

$$\begin{equation}
\begin{array}{rcl}
b_{x,y}^{i} = a_{x,y}^{i}/(k+\alpha \sum_{j=max(0, i-n/2)}^{min(N-1,i+n/2)} (a_{x,y}^{j})^{2})^{\beta}
\end{array}
\end{equation}$$

N是kernel的总数，n是参与一般化的相邻的kernel的数量，$k,n,\alpha,\beta$ 都是hyper-parameters，AlexNet中设置 $k=2,n=5,\alpha=10^{-4},\beta=0.75$。

### 其他一些设置

采用Dropout，随机梯度下降训练，GPU加速训练...

## GoogLeNet

GoogLetNet是相对于[NIN](http://blog.csdn.net/lehyu/article/details/52315206)更稀疏的结构，而且在[ILSVRC2014](http://image-net.org/challenges/LSVRC/2014/)的目标分类与检测上夺冠。GoogLeNet的结构如图所示：

![GoogLeNet](https://raw.githubusercontent.com/Lehyu/lehyu.cn/master/image/DL/CNNs/googlenet.png)

### Inception

与NIN由mlpconv堆积而成类似，GoogLeNet主要有Inception堆积而成。初始版本由下图所示：

![Inception_naive](https://raw.githubusercontent.com/Lehyu/lehyu.cn/master/image/DL/CNNs/Inception_navie.png)

我们可以知道Inception用卷积层代替了原始mlpconv层中的全连接层，这样就可以使结构更稀疏，但是在这种情况下，为了能够并行计算，所以将 *Previous layer* 的所有feature map 全连接到一块进行卷积，因此会导致卷积核过大(假设pervious layer为 $192 \times 28 \times 28$ , 128是feature map的个数，那么卷积核为 $192 \times k \times k$)，那么会在高层Inception中计算量会十分大，因此提出了第二个version

![Inception_v2](https://raw.githubusercontent.com/Lehyu/lehyu.cn/master/image/DL/CNNs/Inception_v2.png)

我们可以看到在 $3\times3$ 和 $5\times5$ 卷积之前添加了 $1\times1$ 的卷积，这样不仅可以降低参数的数量，而且 $1\times1$ 卷积之后做ReLu非线性变换。

### Details

如果想了解GoogLeNet的caffe配置，可以看[这里](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt)，下面稍微介绍一下其他细节

1. GoogLeNet的参数设置如下表：

![parameters](https://raw.githubusercontent.com/Lehyu/lehyu.cn/master/image/DL/CNNs/parameters.png)

2. 由于GoogLeNet是一个22层(不计pooling)的深度结构，因此有可能在BP的时候会出现梯度弥散的情况，所以为了解决这个问题，GoogLeNet增加了两个辅助分类器softmax0、softmax1，在训练的时候，他们的损失误差乘以一个权值(GoogLeNet里设置为0.3)加到整体损失中。在应用的时候，这两个辅助分类器会被丢掉。GoogLeNet的实验表明，只需要一个辅助分类器就可以达到同样的效果(提升0.5%)。

3. 采用SGD，并且设置momentum为0.9；fixed learning rate schedule，每8 epochs学习率下降4%。

## VGG
//todo


## 对比
1. AlexNet

2. GoogLeNet：提高性能——>提高深度与宽度——>过多的参数导致容易过拟合和计算成本过大——>使过滤器稀疏——>Inception
