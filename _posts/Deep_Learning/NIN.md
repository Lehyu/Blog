Network In Network(NIN)模型要由于AlexNet，NIN只用了AlexNet十分之一的参数却比AlexNet更好，所以我们有必要了解一下这个模型的思路。

## 传统CNN模型的局限性
传统的CNN模型是由卷积层与池化层交替组成，而kernel/filter是一个广义线性模型(GLM)，而一般的线性模型只对线性可分的数据/特征有较好的效果，而对非线性可分的数据/特征适用性很差。那么在传统的CNN是可以看做是一个特征提取器，那么由于它的kernel/filter是一个GLM，因此它对线性可分的特征具有很好的效果，但是实际上特征一般都不是线性可分的，那么像AlexNet中又是如何解决这个问题的呢？这些模型是使用over-complete(过完备？非常多的意思吧)的kernel/filter去覆盖所有潜在的特征，可能造成对同一个特征有多个kernel/filter，而下一层要考虑前一层所有特征的组合，从而造成对下一层的额外负担。很显然，越高层的CNN对原始数据越抽象。

## mlpconv
既然传统的GLM存在这样的缺陷，那么是否可以用一个非线性模型来代替GLM从而达到使用非线性数据/特征的效果？NIN提出了一个新的模型mlpconv，该模型使用MLP(multilayer perceptron)代替了传统CNN中的GLM，选择MLP有两个原因：一是MLP本身也是一个深度模型，可以符合[特征复用](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6472238)的思路，并具有非线性的性质；二是MLP也可以用BP算法进行训练。所以对于mlpconv组成的NIN模型，一个是每层可以使用较少的kernel/filter(MLP)，另一个是深度可以比AlexNet更浅。

![L-CNN_VS_Mlpconv](https://raw.githubusercontent.com/Lehyu/lehyu.github.com/master/image/DL/NIN/mlpconv.png)

mlpconv由多层跟着非线性激活函数(ReLu)的全连接层组成，那么对于一个具有n层全连接层的MLP kernel/filter 而言：
$$f_{i,j,k_{1}}^{1} = max({w_{k_{1}}^{1}}^{T} x_{i,j} + b_{k_{1}}, 0)$$
$$\dots$$
$$f_{i,j,k_{n}}^{n} = max({w_{k_{n}}^{n}}^{T} f_{i,j}^{n-1} + b_{k_{n}}, 0)$$

## Global Average Pooling
传统CNN模型在底层执行卷积操作(作为特征提取器)，在高层应用全连接层来进行分类或者其他目的。但是全连接层容易过拟合，不利于整个网络的泛化。因此Hinton提出了[Dropout](http://arxiv.org/pdf/1207.0580v1.pdf)。而NIN的作者提出在CNN模型中应用全局平均池化的方法来代替传统的全连接层，他认为在最后一层mlpconv层会产生一个feature map(这个feature map与分类任务中的类别存在一一对应的关系)，然后将这个feature map经过全局平均池化后再直接传入到softmax layer。这样的话，在全局平均池化层就没有任何需要训练的参数，并且它对卷积结构更加自然的过渡到执行特征图和类别之间的对应关系，而且由于全局平均池化会对空间数据求平均，因此更鲁棒。

## NIN结构

![NIN](https://raw.githubusercontent.com/Lehyu/lehyu.github.com/master/image/DL/NIN/NIN.png)
