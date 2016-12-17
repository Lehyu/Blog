## Machine Learning 步骤

整个机器学习大致可以分为4个步骤：数据分析，数据预处理，模型选择以及训练和优化。比较宏观的流程如下图所示。

![ML flow](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/ML-flow.png?raw=true)

## 栗子：Polynomial Curve Fitting

假设我们现在有一组从 $f(x)=sin(2\pi x)+\epsilon(x)$ 生成的数据，其中 $\epsilon(x)=N(x\vert 0,\beta)$ 表示噪声。那么我们就得到原始数据为

$\mathbb{x}=(x_1,x_2,\dots,x_N)^T,x_i\in(0,1)$，相对应的函数值 $\mathbb{t}=(t_1,t_2,\dots,t_N)^T$。我们要预测，一个新的数据 $\tilde{x}$ 相对应的函数值。

![training data set](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/curve-fitting-points.png?raw=true)

我们得到的只有 $\mathbb{x},\mathbb{t}$ ，并不知道 $f(x)$ 的形式。

### 数据分析(Data Analysis)

由于训练数据由输入向量与其对应的输出向量组成，因此这是一个监督学习(supervised learning)的问题，并且由于输出向量是一组连续的变量，故又可以细分成回归问题(regression)。那么我们可以用以下函数取逼近

$y(x,\mathbb{w})=w_0+w_1x+w_2x^2+\dots+w_Mx^M=\sum_{j=0}^{M}w_jx^j \tag{1}$

[理论]()

### 数据预处理(Preprocess)

数据预处理具有以后优点：

1. 减小输入的多变性，比如输入数据是一系列图片时，我们可能需要将这些图片转换成固定尺寸的图片
2. 加速计算，例如归一化处理，能够输入向量转换到均值为0的特征空间，从而减小划分曲面的偏置搜索
3. 其他

在curve fitting中，我们需要将输入向量转换成 $X=\left[\begin{matrix}1&x_0^1&\dots&x_0^M \\ 1&x_1^1&\dots&x_1^M \\  \vdots & \vdots & \ddots & \vdots \\ 1&x_N^1&\dots&x_N^M \end{matrix}\right]$

如果是在训练的时候计算 $x_i^j$ 的话，会消耗大量的计算资源。

### 模型选择

#### 损失函数

由数据分析步骤，我们已经得到了基础的模型，我们希望我们的预测值 $y(x,\mathbb{w})$ 与 $t$ 尽量相近，那么我们可以用以下度量来表示

$$\begin{equation}
\begin{array}{rcl}
L(y,t) &=& \{y(x,\mathbb{w})-t\}^2\tag{2}
\end{array}
\end{equation}$$

![error function](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/polynomial_curve_fitting_error_function.png?raw=true)

那么对于整体训练数据的度量就可以由(2)得到，

$$\begin{equation}
\begin{array}{rcl}
E(\mathbb{w}) &=& \int_{\mathbb{x}\times\mathbb{t}}{P(x,t)L(y(x,\mathbb{w}),t)}\mathrm{d}x\mathrm{d}t \tag{3}
\end{array}
\end{equation}$$

上式(3)在本例中可以等价于

$$\begin{equation}
\begin{array}{rcl}
E(\mathbb{w}) &=& \frac{1}{N}\sum_{n=1}^NL(y,t)\tag{4}
\end{array}
\end{equation}$$

需要注意的是公式(3)与公式(4)表达的意思并不是一样的，式(3)是期望风险(模型关于联合分布的期望损失)，式(4)是经验风险(模型关于训练样本的平均损失)，当 $P(x,t)=\frac{1}{N}$ 或者样本容量N趋向于无穷时，经验风险趋向于期望风险。

#### 超参数

式(1)里的参数M是不能在训练中直接学到的，需要在训练之前设置，因此称为超参数(hyperparameter)，一个方法是令 $M=1,2,3,\dots$，但是在经验风险下  $E(\mathbb{w})=\frac{1}{2}\sum_{n=1}^NL(y(x,\mathbb{w}),t)$，M的不同会使模型的较大的性能差；另一个方法是采用正则化方法

$$\begin{equation}
\begin{array}{rcl}
\tilde{E}(\mathbb{w}) &=& \frac{1}{2}\sum_{n=1}^NL(y(x,\mathbb{w}),t)+\frac{\lambda}{2}\vert \vert \mathbb{w}\vert \vert ^2\tag{5}
\end{array}
\end{equation}$$

之后会详细介绍这两种损失函数的概率意义。

### 训练与优化

在本例中可以用[最小二乘法]()，直接求解。

### 实验结果

#### E(W)下的结果

![comparison_M](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/comparison_M.png?raw=true)

![generalization](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/generalization.png?raw=true)

我们可以从上图看到，当 $M=0,1,3,9$ 的时候，随着M的增加，曲线能很好更好的拟合训练数据，在 $M=9$ 的时候 $E(W)=0$，但是很明显如果来了一个新的测试数据，$M=3$ 的效果是最好的，这是因为 $M=0,1$ 时，曲线处于欠拟合(underfitting)的状态，而 $M=9$ 时则过拟合(overfitting)。

直觉上，我们认为 $M=9$ 的模型会比 $M<9$ 的模型要好，至少 $M=9$ 的模型的性能不会低于 $M<9$ 的模型，因为 $M=9$ 的模型可以产生 $M<9$ 的特殊子集(如 $M=3$时 令 $w_i=0, i = 4,\dots,9$)。下表是相应的权值 $w^*$。*
![weights](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/weights.png?raw=true)

我们可以看到，随着M的增加，其相应的权值也会变得非常大。但是由于训练数据存在噪声，从而使整个曲线的震荡幅度变大。数据量越大，噪声的影响就越小。

![comparison_N](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/comparison_N_M.png?raw=true)

#### $\tilde{E}(W)$ 下的结果

![comparison_lambda](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/comparison_lambda.png?raw=true)

![generalization_lambda](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/generalization_lambda.png?raw=true)

![weights_lambda](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/weights_lambda.png?raw=true)

由上述三图可以知道，在 $\tilde{E}(W)$ 的情况下，模型能够达到相当好的效果，这是因为正则项能够使权值收缩。从而能够降低噪声造成的震荡幅度。

## 损失函数

### 损失函数的概率意义

![probability_viewpoint](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/probability_viewpoint.png?raw=true)

#### 最大似然

对于训练样本 $(\mathbb{x}, \mathbb{t})$，我们希望 $p(\mathbb{t}\vert \mathbb{x}, \mathbb{w}) =\prod_{i=1}^N N(t_i\vert y(x_i, \mathbb{w}), \beta)$ 达到最大，那么等价于

$$\begin{equation}
\begin{array}{rcl}
\ln{p(\mathbb{t}\vert \mathbb{x}, \mathbb{w})} &=& -\frac{N}{2}\ln{2\pi}+\frac{N}{2}\ln{\beta}-\frac{\beta}{2}\sum_{n=1}^N\{y(x_n, \mathbb{w})-t_n\}^2
\end{array}
\end{equation}\tag{6}$$

即最小化 $E(W)$ 等价于最大化 $\ln{p(\mathbb{t}\vert \mathbb{x}, \mathbb{w})}$，即 $E(W)$ 具有最大似然的性质。

#### 最大后验

前面的假设是对于任意的 $\mathbb{w}$ 具有相同的选择概率，但是如果我们引入先验知识，即

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{w}\vert \alpha) &=& N(\mathbb{w}\vert 0,\alpha^{-1}\boldsymbol{I}) \tag{7}
\end{array}
\end{equation}$$

那么 $p(\mathbb{t},\mathbb{w}\vert \mathbb{x})$ 最大，由于 $p(\mathbb{t})$ 可以看作一个常数，那么 $p(\mathbb{t},\mathbb{w}\vert \mathbb{x})$ 就等价于

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{w}\vert \mathbb{x}, \mathbb{t}, \alpha,\beta)\propto p(\mathbb{t}\vert \mathbb{x},\beta)p(\mathbb{w}\vert \alpha)\tag{8}
\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
\ln\{p(\mathbb{t}\vert \mathbb{x},\beta)p(\mathbb{w}\vert \alpha)\} &=& -\frac{N}{2}\ln{2\pi}+\frac{N}{2}\ln{\beta}-\frac{\beta}{2}\sum_{n=1}^N\{y(x_n, \mathbb{w})-t_n\}^2-\frac{N}{2}\ln{2\pi}+\frac{N}{2}\ln{\alpha}-\frac{\alpha}{2}\vert \vert \mathbb{w}\vert \vert ^2 \tag{9}
\end{array}
\end{equation}$$


即等价于最小化 $\frac{\beta}{2}\sum_{n=1}^N\{y(x_n, \mathbb{w})-t_n\}^2+\frac{\alpha}{2}\vert \vert \mathbb{w}\vert \vert ^2$，也就是最小化 $\tilde{E}(W)$ 等价于最大化后验概率。

### bias和variance

$$\begin{equation}
\begin{array}{rcl}
E(L) &=& \int\int{L(y(\mathbb{x}),t)p(\mathbb{x},t)}\mathrm{d}x\mathrm{d}t \\  &=&  \int\int{\{y(x)-t\}^2p(\mathbb{x},t)}\mathrm{d}x\mathrm{d}t \tag{10}
\end{array}
\end{equation}$$

求极值

$$\begin{equation}
\begin{array}{rcl}
\frac{\partial{E(\boldsymbol{L})}}{\partial{y(\mathbb{x})}} &=& 2\int{\{y(\mathbb{x})-t\}p(\mathbb{x},t)}\mathrm{d}t = 0\tag{11}
\end{array}
\end{equation}$$

那么得到

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x}) &=& \frac{\int{tp(\mathbb{x},t)}\mathrm{d}t}{p(\mathbb{x})}=E_t[t\vert \mathbb{x}]
\end{array}
\end{equation}$$

令

$$\begin{equation}
\begin{array}{rcl}
\{y(\mathbb{x})-t\}^2 &=& \{y(\mathbb{x})-E[t\vert \mathbb{x}]+E[t\vert \mathbb{x}]-t\}^2\\
&=&\{y(\mathbb{x})-E[t\vert \mathbb{x}]\}^2+2\{y(\mathbb{x})-E[t\vert \mathbb{x}]\}^2\{E[t\vert \mathbb{x}]-t\}+\{t-E[t\vert \mathbb{x}]\}^2\tag{12}
\end{array}
\end{equation}$$

将式(12)代入到式(10)，可得到

$$\begin{equation}
\begin{array}{rcl}
E(L) &=& \int{\{y(\mathbb{x})-E[t\vert \mathbb{x}]\}^2p(\mathbb{x})}\mathrm{d}\mathbb{x}+\int{\{t-E[t\vert \mathbb{x}]\}^2p(x,t)}\mathrm{d}x\mathrm{d}t \tag{13}
\end{array}
\end{equation}$$

式(13)中的第一项可以表示为bias，第二项可以表示为variance，$E(L)=bias+variance$，在polynomial curve fitting中，损失函数只优化了bias项，而variance项则相当于固定误差(对于特定的M)。在其他例子中，我们可以看到，函数的优化相当于trade-off between bias and variance。

![bias vs variance](https://github.com/Lehyu/lehyu.github.com/blob/master/image/PRML/chap1/bias_variance.png?raw=true)
