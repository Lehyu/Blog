# 分类的线性模型

分类的目标是在给定输入，预测具有离散性质的目标值。输入空间被多个决策平面划分成多个决策区域，每个区域代表一个类别。决策平面是输入特征的线性函数(待会会详细介绍)，因此在D维空间上的决策平面是(D-1)维的超平面，如果数据能够被这些决策平面准确划分成n个类别区域，那么数据集线性可分(linearly separable)。

当有K(>2)类时，我们采用 1-of-K 编码格式也叫one-hot encoding。 $\mathbb{t}=\{0,\dots,1,\dots,0\}^T,\sum_{k}t_k=1$。

[生成模型与判别模型](生成模型与判别方法.md)

这章的模型可以一般表示为

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x}) &=& f(\mathbb{w}^T\phi(\mathbb{x})+w_0) \\
\end{array}
\end{equation}$$

其中 $f(\cdot)$ 是激活函数。如果令 $f$ 为一个恒等函数(identity function)，即 $f(\cdot)=\cdot$，那么这个模型就变成了[第三章](chapter3_prml.md)的线性回归模型；而如果 $f$ 是非线性函数，那么这个模型就为分类模型，是一个广义线性模型(Generalized Linear Model,GLM)，这是因为决策平面为 $y(\mathbb{x})=\text{constant}\Rightarrow \mathbb{w}^T\phi(\mathbb{x})+w_0=\text{constant}$，可以看到决策平面是输入特征的线性函数。

## 4.1 判别函数(Discriminant Functions)

在给定输入特征时，判别函数输出一个类别 $\mathcal{C}_k$

### 4.1.1 二分类问题

最简单的判别函数为如下的线性判别函数

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x}) &=& \mathbb{w}^T\mathbb{x}+w_0 \\
\end{array}
\end{equation}$$

当 $y(\mathbb{x})\ge 0$ 时，我们将 $\mathbb{x}$ 分到类别 $\mathcal{C}_1$，否则分到 $\mathcal{C}_2$。因此bias项 $w_0$ 的负值有时候也被称为阈值。

因此决策平面 $\mathcal{S}$ 就并定义成了 $y(\mathbb{x})= 0$，从几何上看，我们可以知道 $\mathbb{w}^T$ 是 $\mathcal{S}$ 的法线，那么 $w_0$ 就可就决定了决策平面与原点的距离  $-\frac{w_0}{\|\mathbb{w}\|}$ 。

![geometry of linear discriminant function](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/geometry_of_LDF.png?raw=true)

上图中 $\mathbb{x}$ 是空间上任意一点， $\mathbb{x_\bot}$ 是 $\mathbb{x}$ 正交投影到平面 $\mathcal{S}$ 上的点， $r$ 是 $\mathbb{x}$ 到 $\mathcal{S}$ 的距离，那么

$$\begin{equation}
\begin{array}{rcl}
\mathbb{x} &=& \mathbb{x_\bot}+r\frac{\mathbb{w}}{\|\mathbb{w}\|} \\
r &=& \frac{y(\mathbb{x})}{\|\mathbb{w}\|}
\end{array}
\end{equation}$$

如果令 $x_0=1,\tilde{\mathbb{w}}=\{w_0,\mathbb{w}\},\tilde{\mathbb{x}}=\{x_0,\mathbb{x}\}$，那么 $y(\mathbb{x})=\tilde{\mathbb{w}}^T\tilde{\mathbb{x}}$

### 4.1.2 多类别

对于多类别，可以训练K-1个分类器，每个分类器可以看做是一个二分类问题，即类别 $\mathcal{C}_k$ 与 非 $\mathcal{C}_k$，由于约束，我们训练K-1个分类器即可；训练 $K(K-1)/2$ 个分类器，类别为 $\mathcal{C}_k$ 和 $\mathcal{C}_j$。这两种方法都会导致模糊区域的问题

![ambiguous regions](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/o2o_o2a.png?raw=true)

为了解决模糊区域问题，可以考虑 K 类别判别(K-class discriminant)，

$$\begin{equation}
\begin{array}{rcl}
y_k(\mathbb{x}) &=& \mathbb{w}_k^T\mathbb{x}+w_{k0}
\end{array}
\end{equation}$$

虽然形式上有点类似前两种方法，但是只有当所有 $j\neq k,y_k(\mathbb{x})>y_j(\mathbb{x})$ 时，才分类到 $\mathcal{C}_k$ ，那么 $\mathcal{C}_k$ 与 $\mathcal{C}_j$ 的决策边界(平面) 就变成了 $y_k(\mathbb{x})=y_j(\mathbb{x})\Rightarrow (\mathbb{w}_k-\mathbb{w}_j)^T\mathbb{x}+(w_{k0}-w_{j0})=0$，这与二分类的决策平面一致。

![multiclass linear discriminant](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/multi_LDF.png?raw=true)

由上图知道 $\mathbb{x}_a,\mathbb{x}_b$ 是决策区域 $\mathcal{R}_k$ 的任意两个点，那么在直线 $\mathbb{x}_a\mathbb{x}_b$ 上的任一点 $\hat{\mathbb{x}}$ ，我们可以表示为

$$\begin{equation}
\begin{array}{rcl}
\hat{\mathbb{x}} &=& \lambda\mathbb{x}_a+(1-\lambda)\mathbb{x}_b ,0\leq\lambda\leq1\\
y_k(\hat{\mathbb{x}}) &=& \lambda y_k(\mathbb{x}_a)+(1-\lambda)y_k(\mathbb{x}_b)
\end{array}
\end{equation}$$

明显对于任意 $j\neq k$，我们有 $y_k(\hat{\mathbb{x}})\geq y_j(\hat{\mathbb{x}})$ ，即对于线性可分的数据，$\mathcal{R}_k$ 是单连通凸区域(singly connected and convex)。

### 4.1.3 分类的最小二乘法

采用最小二乘法能够使预测值逼近 $E[\mathbb{t} \vert \mathbb{x}]$，详细参考[第三章](chapter3_prml.md)中的最小二乘法的解释。 对于K类别，我们有

$$\begin{equation}
\begin{array}{rcl}
\mathbb{y}(\mathbb{x}) &=& \tilde{\boldsymbol{W}}^T\tilde{\mathbb{x}} \\
E_D(\tilde{\boldsymbol{W}}) &=& \frac{1}{2}Tr\left\{ (\tilde{\boldsymbol{X}}\tilde{\boldsymbol{W}}-\boldsymbol{T})^T(\tilde{\boldsymbol{X}}\tilde{\boldsymbol{W}}-\boldsymbol{T})   \right\} \\
\Rightarrow \tilde{\boldsymbol{W}} &=& (\tilde{\boldsymbol{X}}\tilde{\boldsymbol{X}})^{-1}\tilde{\boldsymbol{X}}^T\boldsymbol{T}
\end{array}
\end{equation}$$

最小二乘法对离群点不鲁棒，如下图

![robustness of least squares](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/least_squares_robustness.png?raw=true)

左图的决策边界已经能够很好地划分两个区域了，但是来了一些新的数据的时候，如右图，尽管原先的决策边界也能够很好地划分数据，但是由于采用了最小二乘法，为了使损失达到最小，即划分边界距离两个类的条件期望 $E[\mathbb{t} \vert \mathbb{x}]$ 最近，从而驱使原先的决策边界偏离，即右图紫色边界线。

(书上原话: <font color=red>The sum-of-squares error functin penalizes predictions that are 'too correct' in that they lie a long way on the correct side of descision boundary.</font>)

![least_squares_problem](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/least_squares_problem.png?raw=true)

<font color=red>我认为</font>导致左图的原因除了二乘法原因外，还有 K-class discriminant方法中，所有的决策区域必定相交于一个平面，所以左图的数据对于 K-class discriminant是不可分的(?)

导致上述问题的一个根本原因是，最小二乘法是高斯分布假设下的最大似然估计解决方案，而对于分类问题的目标值是离散的，而不是连续的，从而与高斯分布假设不相符。

### 4.1.4 Fisher线性判别(Fisher's linear discriminant)

线性判别函数一般表示为 $y=\mathbb{w}^T\mathbb{x}$， 从几何上看，我们将 D 为输入 $\mathbb{x}$ 经过一个变换之后，输出了一个一维的空间 $y$，并且在这个一维空间上不同类的数据是可分的。考虑2分类问题，只有推广。如果将每个类看成一个簇，那么它中点看做

$$\begin{equation}
\begin{array}{rcl}
\mathbb{m}_1=\frac{1}{N_1}\sum_{n\in\mathcal{C_1}}\mathbb{x}_n &,& \mathbb{m}_2=\frac{1}{N_2}\sum_{n\in\mathcal{C_2}}\mathbb{x}_n
\end{array}
\end{equation}$$

为了使不同类的数据分开，我们使 $\mathbb{m}_1,\mathbb{m}_2$ 投影到一维空间上的距离最远，即

$$\begin{equation}
\begin{array}{rcl}
m_2-m_1 = \mathbb{w}^T(\mathbb{m}_2-\mathbb{m}_1)
\end{array}
\end{equation}$$

假设 $\mathbb{x}_a\in\mathcal{C}_1$，当 $\mathbb{x}_a$ 在 $\mathbb{m}_1$ 附近或者 $\overrightarrow{m_1x_a}$ 的方向与 $\overrightarrow{m_1m_2}$ 背离(夹角大于90度)，那么 $\mathbb{x}_a$ 能够与 $\mathcal{C}_2$ 中的点很好地分离，同理 $\mathcal{C}_2$。
但是对于 $\mathbb{x}_a$ 不在 $\mathbb{m}_1$ 附近并且 $\overrightarrow{m_1x_a}$ 的方向与 $\overrightarrow{m_1m_2}$ 同向(不是夹角等于0，而是夹角小于90度) ，那么这些点就有可能不能很好地分离开，如下图左图所示

![Fisher discriminant analysis](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/FDA.png?raw=true)

[Fisher linear discriminant wiki](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher.27s_linear_discriminant)。Fisher线性判别对这个问题进行了研究，他的思想要使类内(within-class)数据的方差最小，并且使类间(between-class)数据的方差最大，从而得出了Fisher criterion

$$\begin{equation}
\begin{array}{rcl}
J(\mathbb{w}) &=& \frac{(m_2-m_1)^2}{s_1^2+s_2^2} \\
s_k^2 &=& \sum_{n\in\mathcal{C}_k}(y_n-m_k)^2
\end{array}
\end{equation}$$

经过调整后，我们就得到了上图右图，可以看出此时不同类的数据已经能够很好地划分开来。

求解 $\mathbb{w}$

$$\begin{equation}
\begin{array}{rcl}
J(\mathbb{w}) &=& \frac{\mathbb{w}^T\boldsymbol{S}_B\mathbb{w}}{\mathbb{w}^T\boldsymbol{S}_W\mathbb{w}} \\
\boldsymbol{S}_B &=& (\mathbb{m}_2-\mathbb{m}_1)(\mathbb{m}_2-\mathbb{m}_1)^T \\
\boldsymbol{S}_W &=& \sum_{n\in\mathcal{C}_1}(\mathbb{x}_n-\mathbb{m}_1)(\mathbb{x}_n-\mathbb{m}_1)^T+\sum_{n\in\mathcal{C}_2}(\mathbb{x}_n-\mathbb{m}_2)(\mathbb{x}_n-\mathbb{m}_2)^T \\
\overset{\frac{\partial J}{\partial\mathbb{w}}=0}\Rightarrow (\mathbb{w}^T\boldsymbol{S}_B\mathbb{w})\boldsymbol{S}_W\mathbb{w} &=& \boldsymbol{S}_B\mathbb{w} (\mathbb{w}^T\boldsymbol{S}_W\mathbb{w}) \\
\Rightarrow \mathbb{w} &\propto& \boldsymbol{S}_W^{-1}(\mathbb{m}_2-\mathbb{m}_1)
\end{array}
\end{equation}$$

需要注意的是此时求解的 $\mathbb{w}$ 是决策超平面的法向量，还需要选取一个阈值 $y_0$。

### 4.1.5 Fisher判别与最小二乘法的关系

最小二乘法：驱使模型的预测值与目标值尽可能相近

Fisher判别：最大化输出空间的类别差别

在二分类问题中，Fisher判别可以当做最小二乘法的一个特殊情况。在最小二乘法中，当我们用 $\{N/N_1,-N/N_2\}$ 代替 \{1,0\} 时，即可得到Fisher判别。

### 4.1.6 多类别的Fisher判别

采用K-class discriminant模型，我们有

$$\begin{equation}
\begin{array}{rcl}
\mathbb{y} = \boldsymbol{W}^T\mathbb{x}
\end{array}
\end{equation}$$

类比二分类的Fisher判别，我们有

$$\begin{equation}
\begin{array}{rcl}
\mathbb{m}_k &=& \frac{1}{N_k}\sum_{n\in\mathcal{C}_k}\mathbb{x}_n \\
\boldsymbol{S}_k &=& \sum_{n\in\mathcal{C}_k}(\mathbb{x}_n -\mathbb{m}_k)(\mathbb{x}_n -\mathbb{m}_k)^T \\
\mathbb{m} &=& \frac{1}{N}\sum_{n}\mathbb{x}_n = \frac{1}{N}\sum_{k=1}^KN_k\mathbb{m}_k \\
\boldsymbol{S}_B &=& \sum_{k=1}^KN_k(\mathbb{m}_k-\mathbb{m})(\mathbb{m}_k-\mathbb{m})^T \\
\boldsymbol{S}_W &=& \sum_{k=1}^K\boldsymbol{S}_k \\
J(\boldsymbol{W}) &=& \text{Tr}\left\{ (\boldsymbol{W}\boldsymbol{S}_W\boldsymbol{W}^T)^{-1}(\boldsymbol{W}\boldsymbol{S}_B\boldsymbol{W}^T)  \right\}
\end{array}
\end{equation}$$

### 4.1.7 感知器算法

感知器算法中与前两种方法不同的是采用了step激活函数

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x}) &=& f(\mathbb{w}^T\phi(\mathbb{x})) \\
f(a) &=& \cases{+1,a\ge0 \\-1,a<0}
\end{array}
\end{equation}$$

其中 $\phi_0(\mathbb{x})=1$，而且跟前面的方法目标值的表达不同 $t\in\{-1,+1\}$，而不是 $t\in\{0,1\}$，若 $\mathbb{x}\in\mathcal{C}_1,t=+1$，否则 $t=-1$。因此对于分类正确的数据 $(\mathbb{x}_n,t_n),y_nt_n>0$，对于误分的数据 $y_nt_n<0$，所以感知器算法的损失函数就采用了

$$\begin{equation}
\begin{array}{rcl}
E(\mathbb{w}) &=& -\sum_{n\in\mathcal{M}}\mathbb{w}^T\phi(\mathbb{x}_n)t_n
\end{array}
\end{equation}$$

$\mathcal{M}$ 是误分的集合，采用随机梯度下降法对上述损失函数求解。

![perceptron algorithm](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/perceptron_learning_algorithm.png?raw=true)

需要注意的是感知器只适用于二分类问题，而且当数据是非线性可分时，迭代不收敛；而且就算对于线性可分的数据，由于每一次迭代都会更改权值，从而影响先前的数据，使原本分类正确的数据在下一轮分类错误，迭代次数增大(当然最后是肯定收敛的)。因此在训练的时候，我们不知道数据到底是非线性可分还是迭代次数不够。

## 4.2 概率生成模型(Probabilistic Genertive Models)

首先对 $p(\mathbb{x}\vert\mathcal{C}_k),p(\mathcal{C}_k)$ 建模，然后根据贝叶斯定理计算 $p(\mathcal{C}_k\vert\mathbb{x})$，对于二分类问题

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{C}_1\vert\mathbb{x}) &=& \frac{p(\mathbb{x}\vert\mathcal{C}_1)p(\mathcal{C}_1)}{p(\mathbb{x}\vert\mathcal{C}_1)p(\mathcal{C}_1)+p(\mathbb{x}\vert\mathcal{C}_2)p(\mathcal{C}_2)} \\
&=&\frac{1}{1+\exp(-a)} \\
a &=& \ln\frac{p(\mathbb{x}\vert\mathcal{C}_1)p(\mathcal{C}_1)}{p(\mathbb{x}\vert\mathcal{C}_2)p(\mathcal{C}_2)}
\end{array}
\end{equation}$$

对于多类别问题

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{C}_k\vert\mathbb{x}) &=& \frac{p(\mathbb{x}\vert\mathcal{C}_k)p(\mathcal{C}_k)}{\sum_{j}p(\mathbb{x}\vert\mathcal{C}_j)p(\mathcal{C}_j)} \\
&=&\frac{\exp(a_k)}{\sum_{j}\exp(a_j)} \\
a_k &=& \ln p(\mathbb{x}\vert\mathcal{C}_k)p(\mathcal{C}_k)
\end{array}
\end{equation}$$


### 4.2.1 连续型特征

对于连续型变量，我们可以假设似然函数服从高斯分布，并且所有的类别具有相同的协方差矩阵

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{x}\vert\mathcal{C}_k) &=& \mathcal{N}(\mathbb{x}\vert \mathbb{\mu}_k, \Sigma)
\end{array}
\end{equation}$$

对于二分类问题，则有

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{C}_1\vert\mathbb{x}) &=& \sigma(\mathbb{w}^T\mathbb{x}+w_0) \\
\mathbb{w} &=& \Sigma^{-1}(\mathbb{\mu}_1-\mathbb{\mu}_2) \\
w_0 &=& -\frac{1}{2}\mathbb{\mu}_1^T\Sigma^{-1}\mathbb{\mu}_1+\frac{1}{2}\mathbb{\mu}_2^T\Sigma^{-1}\mathbb{\mu}_2+\ln\frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}
\end{array}
\end{equation}$$

由上面的推导我们知道，先验概率只出现在bias项，因此改变先验会影响决策平面的偏移。

对于一般K类，则有

$$\begin{equation}
\begin{array}{rcl}
a_k(\mathbb{x}) &=& \mathbb{w}_k^T\mathbb{x}+w_{k0} \\
\mathbb{w}_k &=& \Sigma^{-1}\mathbb{\mu}_k \\
w_{k0} &=& -\frac{1}{2}\mathbb{\mu}_k^T\Sigma^{-1}\mathbb{\mu}_k+\ln p(\mathcal{C}_k)
\end{array}
\end{equation}$$

上述推导都可以通过贝叶斯定理与替换推出。

### 4.2.2 最大似然方案

对于二分类问题，假设我们有训练数据 $\{\mathbb{x}_n,t_n\},n=1,\dots,N$，当 $\mathbb{x}_n\in\mathcal{C}_1,t_n=1;\mathbb{x}_n\in\mathcal{C}_2,t_n=0，p(\mathcal{C}_1)=\pi,p(\mathcal{C}_2)=1-\pi$ 。那么

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{x}_n,\mathcal{C}_1) &=& p(\mathcal{C}_1)p(\mathbb{x}_n\vert \mathcal{C}_1) = \pi\mathcal{N}(\mathbb{x}\vert \mathbb{\mu}_1,\Sigma) \\
p(\mathbb{x}_n,\mathcal{C}_20) &=& p(\mathcal{C}_2)p(\mathbb{x}_n\vert \mathcal{C}_2) = (1-\pi)\mathcal{N}(\mathbb{x}\vert \mathbb{\mu}_2,\Sigma) \\
\Rightarrow p(\mathbb{t}\vert \pi,\mathbb{\mu}_1,\mathbb{\mu}_2,\Sigma) &=& \prod_{n=1}^N[p(\mathcal{C}_1\vert\mathbb{x}_n)]^{t_n}[p(\mathcal{C}_2\vert\mathbb{x}_n)]^{1-t_n} \\
&\propto& \prod_{n=1}^N[\pi\mathcal{N}(\mathbb{x}\vert \mathbb{\mu}_1,\Sigma)]^{t_n}[(1-\pi)\mathcal{N}(\mathbb{x}\vert \mathbb{\mu}_2,\Sigma)]^{1-t_n}
\end{array}
\end{equation}$$

为了方便，我们令 $p(\mathbb{t}\vert \pi,\mathbb{\mu}_1,\mathbb{\mu}_2,\Sigma) =\prod_{n=1}^N[\pi\mathcal{N}(\mathbb{x}\vert \mathbb{\mu}_1,\Sigma)]^{t_n}[(1-\pi)\mathcal{N}(\mathbb{x}\vert \mathbb{\mu}_2,\Sigma)]^{1-t_n}$，根据 $\pi,\mathbb{\mu}_1,\mathbb{\mu}_2$ 求其偏导为0的解

$$\begin{equation}
\begin{array}{rcl}
\pi &=& \frac{N_1}{N_1+N_2} \\
\mathbb{\mu}_1 &=& \frac{1}{N_1}\sum_{n\in\mathcal{C}_1}\mathbb{x}_n = \frac{1}{N_1}\sum_{n=1}^Nt_n\mathbb{x}_n \\
\mathbb{\mu}_2 &=& \frac{1}{N_2}\sum_{n=1}^N(1-t_n)\mathbb{x}_n \\
\Sigma &=& \frac{N_1}{N}\boldsymbol{S}_1+\frac{N_2}{N}\boldsymbol{S}_2 \\
\boldsymbol{S}_k &=& \frac{1}{N_k}\sum_{n\in\mathcal{C}_k}(\mathbb{x}_n-\mathbb{\mu}_k)(\mathbb{x}_n-\mathbb{\mu}_k)^T
\end{array}
\end{equation}$$

### 4.2.3 离散型特征

假设输入特征为离散型，我们考虑 $x_i\in\{0,1\}$ 的情况，对于D维的特征向量，我们假设每一维特征都相互独立，则有

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{x} \vert \mathcal{C}_k) &=& \prod_{j=1}^D\mu_{kj}^{x_j}(1-\mu_{kj})^{1-x_j} \\
a_k &=& \ln p(\mathbb{x}\vert \mathcal{C}_k)p(\mathcal{C}_k) \\
&=& \sum_{j=1}^D\{x_j\ln\mu_{kj}+(1-x_j)\ln(1-\mu_{kj})\}
\end{array}
\end{equation}$$

采用最大似然估计的分析跟连续型特征的类似，所以不再复述。

### 4.2.4 指数族

由[第二章](chapter2_prml.md)我们知道高斯分布与伯努利分布都可以划分到指数族中，现在我们泛化

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{x} \vert \boldsymbol{\lambda}_k) &=& h(\mathbb{x})g(\boldsymbol{\lambda}_k)\exp(\boldsymbol{\lambda}_k\mathbb{u}(\mathbb{x}))
\end{array}
\end{equation}$$

如果我们令 $\mathbb{u}(\mathbb{x})=\mathbb{x}$，并且再对 $\mathbb{x}$ 缩放，那么

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{x} \vert \boldsymbol{\lambda}_k,s) &=& \frac{1}{s} h( \frac{1}{s}\mathbb{x})g(\boldsymbol{\lambda}_k)\exp( \frac{1}{s}\boldsymbol{\lambda}_k\mathbb{u}(\mathbb{x}))  \\
a_k(x) &=& \boldsymbol{\lambda}_k^T\mathbb{x}+\ln g(\boldsymbol{\lambda}_k)+\ln p(\mathcal{C}_k)
\end{array}
\end{equation}$$

## 4.3 概率判别模型

在上一节中我们引入了 $a_k(\mathbb{x})$ 在生成模型中好像有点多余，那是因为是为了给概率判别模型做铺垫。在生成模型中，我们必须先要对 $p(\mathbb{x}\vert \mathcal{C}_k),p(\mathcal{C}_k)$ 建模才能得到后验概率，而我们已经知道 $y(\mathbb{x})=f(\mathbb{w}^T\phi(\mathbb{x}))$ 是一个广义的线性模型，并且如果选择一个特殊的激活函数，如sigmoid、softmax，会得到后验概率，从而我们希望直接定义 $p(\mathcal{C}_k\vert \mathbb{x})$ 的函数来求解。概率判别模型的一个有点是参数少。

### 4.3.1 基函数

上面讨论的时候，我们都是考虑原始空间上的分类问题，但是在原始空间上的数据可能非线性可分的，但是如果我们经过一个非线性基函数变换后，在变换的空间上，数据是线性可分的。如下图所示

![nonlinear transform](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/basis_functions.png?raw=true)

在实际问题中，类别条件密度 $p(\mathbb{x}\vert \mathcal{C}_k)$ 会有重叠，因此相应的 $p(\mathcal{C}_k\vert\mathbb{x})$ 有一些 $(0,1)$ 区间的模糊集，这时需要采用decision theory。需要注意的是，非线性变换并不能解决这个问题，甚至有可能引入新的重叠，但是合适的基函数能够使后验概率的建模过程更加简单。

### 4.3.2 逻辑斯蒂回归(Logistic Regression)

对于二分类问题，上一小节我们推导出

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{C}_1\vert \mathbb{x}) &=& y(\mathbb{x}) = \sigma(\mathbb{w}^T\boldsymbol\phi) \\
\sigma(a) &=& \frac{1}{1+\exp(-a)}
\end{array}
\end{equation}$$

那么 $p(\mathcal{C}_2\vert \mathbb{x})=1-p(\mathcal{C}_1\vert \mathbb{x})$

![sigmoid vs probit](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/sigmoid_vs_probit.png?raw=true)

假设 $\boldsymbol\phi$ 的维度是M，那么这个模型的参数只需要M个。而对于生成模型，则需要 $2M+M(M+1)/2=M(M+5)/2$ 个参数。

对于训练集 $\{\boldsymbol\phi_n,t_n\},t_n\in\{0,1\},\boldsymbol\phi_n=\boldsymbol\phi(\mathbb{x}_n),n=1,\dots,N$ ，那么似然函数为

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{t} \vert \mathbb{w}) &=& \prod_{n=1}^Ny_n^{t_n}\{1-y_n\}^{1-t_n} \\
E(\mathbb{w}) &=& -\ln p(\mathbb{t} \vert \mathbb{w}) = -\sum_{n=1}^N\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\}
\end{array}
\end{equation}$$

上式中的 $y_n=\sigma(\mathbb{w}^T\boldsymbol\phi_n)$， $E(\mathbb{w})$ 称为交叉熵损失函数。

$$\begin{equation}
\begin{array}{rcl}
\nabla E(\mathbb{w}) &=& \sum_{n=1}^N(y_n-t_n)\boldsymbol\phi_n
\end{array}
\end{equation}$$

可以采用随机梯度下降求解。

<font color=red>关于最大似然导致严重过拟合的原因没看明白(todo)</font>

### 4.3.3 迭代再加权最小二乘(iterative reweighted least square)

由于sigmoid函数， $\nabla E(\mathbb{w})$ 不再具有闭式解(closed-form solution)。但是它还是一个凸函数，因此具有一个全局最小解，可以采用[Newton-Raphson法](todo)来求解

$$\begin{equation}
\begin{array}{rcl}
\mathbb{w}^{\tau+1}=\mathbb{w}^{\tau}-\boldsymbol{H}\nabla E(\mathbb{w})
\end{array}
\end{equation}$$

在线性回归中

$$\begin{equation}
\begin{array}{rcl}
\nabla E(\mathbb{w}) &=& \sum_{n=1}^N(\mathbb{w}^T\boldsymbol\phi_n-t_n)\boldsymbol\phi_n=\boldsymbol\Phi^T\boldsymbol\Phi\mathbb{w}-\boldsymbol\Phi^T\mathbb{t} \\
\boldsymbol{H} &=& \nabla\nabla E(\mathbb{w}) = \boldsymbol\Phi^T\boldsymbol\Phi \\
\Rightarrow \mathbb{w}^{\tau+1}&=&\mathbb{w}^{\tau}-\boldsymbol{H}\nabla E(\mathbb{w})=(\boldsymbol\Phi^T\boldsymbol\Phi)^{-1}\boldsymbol\Phi^T\mathbb{t}
\end{array}
\end{equation}$$

我们可以看出标准最小二乘法与Newton-Raphson法得到相同的解。

对于logistics回归模型

$$\begin{equation}
\begin{array}{rcl}
\nabla E(\mathbb{w}) &=& \sum_{n=1}^N(y_n-t_n)\boldsymbol\phi_n=\boldsymbol\Phi^T(\mathbb{y}-\mathbb{t}) \\
\boldsymbol{H} &=& \nabla\nabla E(\mathbb{w}) = \sum_{n=1}^N y_n(1-y_n)\boldsymbol\phi_n\boldsymbol\phi_n^T=\boldsymbol\Phi^T\boldsymbol{R}\boldsymbol\Phi \\
\boldsymbol{R} &=& dialog\{y_n(1-y_n)\}\\
\end{array}
\end{equation}$$

由于 $0<y_n<1$，那么对于任意向量 $\mathbb{u},\mathbb{u}^T\boldsymbol{R}\mathbb{u}>0$，即 $\boldsymbol{R}$ 是正定矩阵。所以 $E(\mathbb{w})$ 是一个凸函数。用Newton-Raphson法对logistics模型求解得

$$\begin{equation}
\begin{array}{rcl}
\mathbb{w}^{\tau+1} &=&\mathbb{w}^{\tau}-(\boldsymbol\Phi^T\boldsymbol{R}\boldsymbol\Phi)^{-1}\boldsymbol\Phi^T(\mathbb{y}-\mathbb{t}) \\
&=& (\boldsymbol\Phi^T\boldsymbol{R}\boldsymbol\Phi)^{-1}\boldsymbol\Phi^T\boldsymbol{R}\mathbb{z} \\
\mathbb{z} &=& \boldsymbol\Phi\mathbb{w}^{\tau}-\boldsymbol{R}^{-1}(\mathbb{y}-\mathbb{t})
\end{array}
\end{equation}$$

因为 $\boldsymbol{R}$ 不是常数，而依赖于 $\mathbb{w}$ ，因此我们需要用上式迭代求解。 $\boldsymbol{R}$ 可以看做是方差。

### 4.3.4 多类别逻辑斯蒂回归

4.2小节我们推导出，K分类的一般形式为

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{C}_k\vert \boldsymbol\phi) &=& y_k(\boldsymbol\phi)=\frac{\exp(a_k)}{\sum_{j}\exp(a_j)} \\
a_k &=& \mathbb{w}^T\boldsymbol\phi \\
\frac{\partial y_k}{\partial a_j}  &=& y_k(I_{kj}-y_j)
\end{array}
\end{equation}$$

那么似然函数为

$$\begin{equation}
\begin{array}{rcl}
p(\boldsymbol{T}\vert \mathbb{w}_1, \dots ,\mathbb{w}_K) &=& \prod_{n=1}^N\prod_{k=1}^Kp(\mathcal{C}_k\vert \boldsymbol\phi_n)^{t_{nk}} \\
E(\mathbb{w}_1, \dots ,\mathbb{w}_K) &=& -\ln p(\boldsymbol{T}\vert \mathbb{w}_1, \dots ,\mathbb{w}_K) = -\sum_{n=1}^N\sum_{k=1}^K t_{nk}\ln y_{nk}
\end{array}
\end{equation}$$

其中 $y_{nk}=y_k(\boldsymbol\phi_n)$，那么

$$\begin{equation}
\begin{array}{rcl}
\nabla_{w_j}E(\mathbb{w}_1, \dots ,\mathbb{w}_K) &=&  \sum_{n=1}^N(y_{nj}-t_nj)\boldsymbol\phi_n
\end{array}
\end{equation}$$

我们可以采用随机梯度下降求解 $\mathbb{w}_k^{\tau+1}=\mathbb{w}_k^{\tau}-\eta\nabla_{w_k}E(\mathbb{w}_1, \dots ,\mathbb{w}_K)$，或者考虑IRLS

$$\begin{equation}
\begin{array}{rcl}
\nabla_{w_k}\nabla_{w_j}E(\mathbb{w}_1, \dots ,\mathbb{w}_K) &=&  -\sum_{n=1}^Ny_{nk}(I_{kj}-y_nj)\boldsymbol\phi_n\boldsymbol\phi_n^T
\end{array}
\end{equation}$$

### 4.3.5 Probit Regression

前面我们考虑的激活函数都是logistics或者softmax函数，现在我们考虑其他的激活函数 $p(t\vert a)=f(a)$，当 $a>\theta,t_n=1$，那么 $\theta$ 的任意概率密度 $p(\theta)$，我们得到

$$\begin{equation}
\begin{array}{rcl}
f(a) = \int_{-\infty}^ap(\theta)d\theta
\end{array}
\end{equation}$$

![motivation of probit funtion](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/general_activation_funtion.png?raw=true)

当我们假设 $p(\theta)=\mathcal{N}(\theta\vert 0,1)$ 时，我们有 $\Phi(a) = \int_{-\infty}^a\mathcal{N}(\theta\vert 0,1)d\theta$，$\Phi(a)$就被称为probit函数。

$$\begin{equation}
\begin{array}{rcl}
erf(a) &=&\frac{2}{\sqrt{\pi}}\int_0^a\exp(-\theta^2/2)d\theta \\
\Phi &=& \frac{1}{2}\left\{ 1+\frac{1}{\sqrt2}erf(a) \right\}
\end{array}
\end{equation}$$

之后的求解过程与logistics模型一致，但是需要注意的是probit模型比logistics模型对离群点更敏感。

### 4.3.6 Canonical link functions

## 4.4 拉普拉斯逼近(The Laplace Approximation)

在线性回归的时候我们讨论了贝叶斯分析的形式如下

$$\begin{equation}
\begin{array}{rcl}
p(t\vert \mathbb{t},\alpha,\beta) &=& \int p(t\vert \mathbb{w},\beta)p(\mathbb{w}\vert \mathbb{t},\alpha,\beta) d\mathbb{w} \\
\end{array}
\end{equation}$$

但是在logistics回归中，后验概率不再是高斯分布函数，因此我们希望能够寻求一个高斯分布来近似后验分布。

对于任意一个分布，我们有

$$\begin{equation}
\begin{array}{rcl}
p(z)=\frac{1}{Z}f(z),Z=\int f(z)dz
\end{array}
\end{equation}$$

Laplace逼近的目标就是找到一个高斯函数 $q(z)$ ，使其均值为 $p(z)$ 的模。第一步是找到模 $z_0$，由于高斯分布的对数是变量的二项式函数，那么 $\ln f(z)$ 在 $z_0$ 附近的泰勒展开式如下

$$\begin{equation}
\begin{array}{rcl}
\ln f(z) &\simeq& \ln f(z_0)-\frac{1}{2}A(z-z_0)^2 \\
A &=& -\frac{d^2}{dz^2}\ln f(z)\Big|_{z=z_0}\\
\Rightarrow f(z) &\simeq& f(z_0)\exp\left\{-\frac{A}{2}(z-z_0)^2 \right\}\\
\Rightarrow q(z) &=& \left(\frac{A}{2\pi}\right)^{1/2}\exp\left\{-\frac{A}{2}(z-z_0)^2 \right\}
\end{array}
\end{equation}$$

![Laplace Approximation](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap4/laplace_approximation.png?raw=true)
同理对于M为变量也是如此求解
$$\begin{equation}
\begin{array}{rcl}
\ln f(\mathbb{z}) &\simeq& \ln f(\mathbb{z}_0)-\frac{1}{2}(\mathbb{z}-\mathbb{z}_0)^T\boldsymbol{A}(\mathbb{z}-\mathbb{z}_0) \\
A &=& -\frac{d^2}{d\mathbb{z}^2}\ln f(\mathbb{z})\Big|_{\mathbb{z}=\mathbb{z}_0}\\
\Rightarrow f(\mathbb{z}) &\simeq& f(\mathbb{z}_0)\exp\left\{-\frac{1}{2}(\mathbb{z}-\mathbb{z}_0)^T\boldsymbol{A}(\mathbb{z}-\mathbb{z}_0) \right\}\\
\Rightarrow q(\mathbb{z}) &=& \frac{\vert A\vert^{1/2}}{(2\pi)^{M/2}}\exp\left\{-\frac{1}{2}(\mathbb{z}-\mathbb{z}_0)^T\boldsymbol{A}(\mathbb{z}-\mathbb{z}_0)\right\}
\end{array}
\end{equation}$$

laplace逼近只适用于实数变量。当数据集很大时，Laplace逼近有很好地效果。

### 4.4.1 模型比较

对于模型 $\mathcal{M}_i$ 有参数集 $\{\boldsymbol\theta_i\}$，那么这个模型的model evidence为

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{D}) &=& \int p(\mathcal{D}\vert \boldsymbol\theta)p(\boldsymbol\theta)d\boldsymbol\theta \\
\ln p(\mathcal{D}) &=& \ln p(\mathcal{D}\vert \boldsymbol\theta_{MAP}) +\ln p(\boldsymbol\theta)+\frac{M}{2}\ln2\pi-\frac{1}{2}\ln\boldsymbol{A}
\end{array}
\end{equation}$$

如果我们假设高斯先验概率的参数是平(broad)的话，那么

$$\begin{equation}
\begin{array}{rcl}
\ln p(\mathcal{D}) &=& \ln p(\mathcal{D}\vert \boldsymbol\theta_{MAP}) -\frac{1}{2}M\ln N
\end{array}
\end{equation}$$

## 4.5 Bayesian Logistic Regression

### 4.5.1 拉普拉斯逼近

高斯先验 $p(\mathbb{w})=\mathcal{N}(\mathbb{w}\vert \mathbb{m}_0,\boldsymbol{S}_0)$，那么后验为

$$\begin{equation}
\begin{array}{rcl}
\ln p(\mathbb{w}\vert \mathbb{t}) = -\frac{1}{2}(\mathbb{w}-\mathbb{m}_0)^T\boldsymbol{S}_0^{-1}(\mathbb{w}-\mathbb{m}_0) + \sum_{n=1}^N\{ t_n\ln y_n+(1-t_n)\ln(1-y_n)\} +const
\end{array}
\end{equation}$$

为了获得一个近似后验概率的高斯分布，我们首先求上式的模，即 $\mathbb{w}_{MAP}$，然后求后验概率对数在  $\mathbb{w}_{MAP}$ 的二阶导，得

$$\begin{equation}
\begin{array}{rcl}
\boldsymbol{S}_N &=& -\nabla\nabla\ln p(\mathbb{w}\vert \mathbb{t}) = \boldsymbol{S}_0^{-1}+\sum_{n=1}^Ny_n(1-y_n)\boldsymbol\phi_n\boldsymbol\phi_n^T \\
\Rightarrow q(\mathbb{w}) &=& \mathcal{N}(\mathbb{w}\vert\mathbb{w}_{MAP},\boldsymbol{S}_N)
\end{array}
\end{equation}$$

### 4.5.2 预测分布

上一小节我们已经采用Laplace逼近找到一个高斯分布近似后验概率分布。那么预测分布为

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{C}_1\vert \boldsymbol\phi,\mathbb{t}) &=& \int p(\mathcal{C}_1\vert \boldsymbol\phi,\mathbb{w})p(\mathbb{w}\vert \mathbb{t}) d\mathbb{w}=\int \sigma(\mathbb{w}^T\boldsymbol\phi)q(\mathbb{w})d\mathbb{w}\\
\sigma(\mathbb{w}^T\boldsymbol\phi) &=& \int \delta_a(\mathbb{w}^T\boldsymbol\phi)\sigma(a)da \\
\Rightarrow p(\mathcal{C}_1\vert \boldsymbol\phi,\mathbb{t}) &=& \int \sigma(a)p(a)da \\
p(a)&=& \int \delta_a(\mathbb{w}^T\boldsymbol\phi)q(\mathbb{w})d\mathbb{w}
\end{array}
\end{equation}$$

我们知道 $q(\mathbb{w})$ 是高斯分布，从而 $p(a)$ 也服从高斯分布

$$\begin{equation}
\begin{array}{rcl}
\mu_a &=& E[a]=\int p(a)ada = \int q(\mathbb{w})\mathbb{w}^T\boldsymbol\phi d\mathbb{w} = \mathbb{w}_{MAP}\boldsymbol\phi \\
\sigma_a^2 &=& var[a] = \int p(a)(a-\mu_a)da=\int q(\mathbb{w})\{(\mathbb{w}^T\boldsymbol\phi)^2 - (\mathbb{m}_N^T\boldsymbol\phi)^2\} = \boldsymbol\phi^T\boldsymbol{S}_N\boldsymbol\phi\\
\Rightarrow p(\mathcal{C}_1\vert \boldsymbol\phi,\mathbb{t}) &=& \int \sigma(a)\mathcal{N}(a\vert \mu_a,\sigma_a^2)da
\end{array}
\end{equation}$$

此时由于 $\sigma(a)$ 是logistics sigmoid，因此上述公式还是不可评估的，因此我们又用 $\sigma(a)\simeq \Phi(\lambda a),\lambda^2=\pi/8$ 去逼近。那么上式就等于

$$\begin{equation}
\begin{array}{rcl}
\int \Phi(\lambda a)\mathcal{N}(a\vert \mu,\sigma)da &=& \Phi\left(\frac{\mu}{(\lambda^{-2}+\sigma^2)^{1/2}}\right)\simeq \sigma(\kappa(\sigma^2)a) \\
\kappa(\sigma^2) &=& (1+\pi\sigma^2/8)^{-1/2} \\
\Rightarrow p(\mathcal{C}_1\vert \boldsymbol\phi,\mathbb{t}) &=& \sigma(\kappa(\sigma_a^2)\mu_a)
\end{array}
\end{equation}$$



$$\begin{equation}
\begin{array}{rcl}
\\
1\\
1\\
1\\\
1\\1
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
1\\
\end{array}
\end{equation}$$
