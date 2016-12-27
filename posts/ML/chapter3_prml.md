# 线性回归

回归的目标是在给定输入的情况下，预测具有连续性质的目标值。线性回归中的线性是相对于参数而言的。

## 3.1 线性基函数模型(Linear Basis Function Models)

最简单的线性回归模型是：$y(\mathbb{x},\mathbb{w})=w_0+w_1x_1+\dots+w_Dx_D$ ，很明显这个模型不足以表达复杂的模型，但是我们能够从这个模型中得出线性回归模型的一般形式

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x},\mathbb{w}) = w_0+\sum_{j=1}^Mw_j\phi_j(\mathbb{x})
\end{array}
\end{equation} \tag{1}$$

其中 $\phi_j(\mathbb{x})$ 即基函数，该函数可以是任意的函数，一般为非线性函数(为了提高模型的表达能力)；$w_0$ 为偏置，假设我们令 $\phi_0(\mathbb{x})=1$ ，那么上式就可以简化成

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x},\mathbb{w}) = \sum_{j=0}^Mw_j\phi_j(\mathbb{x})=\mathbb{w}^T\boldsymbol{\phi}(\mathbb{x})
\end{array}
\end{equation}$$

整个模型对于输入是非线性的，而对于参数是线性的，这样就在提高模型表达能力的同时，也简化了模型。但是这种简化也导致了明显的限制，后面会详细介绍。

[第一章](chapter1_prml.md)中的曲线拟合，我们令 $\phi_j(x)=x^j$ ，多项式基函数是输入变量的全局函数，如果一个输入变量的区域改变会影响其他的输入区域，比如 $(2,1,1,1)\to(2,1,1,9)$，但是如果采用如高斯基函数等局部函数的话，就不会出现这种情况。

常见的几类基函数:

1. 多项基函数： $\phi_j(x)=x^j$

2. 高斯基函数： $\phi_j(x)=\exp\left\{-\frac{(x-\mu_j)^2}{2s^2}\right\}$

3. sigmoid：$\phi_j(x)=\sigma\left(\frac{x-\mu_j}{s}\right),\sigma(a)=\frac{1}{1+\exp(-a)}$

![basis_function](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/basis_function.png?raw=true)

### 3.1.1 最大似然和最小二乘法

假设目标值t由判别函数与一个额外的噪声给出: $t=y(\mathbb{x},\mathbb{w})+\epsilon$， 其中噪声为一个均值为0、精度为 $\beta$ 的高斯噪声。那么

$$\begin{equation}
\begin{array}{rcl}
p(t\vert \mathbb{x},\mathbb{w},\beta)=\mathcal{N}(t\vert y(\mathbb{x},\mathbb{w}),\beta^{-1})
\end{array}
\end{equation}$$

假设我们令其损失函数为平方损失函数(square loss function)，那么最优预测值就与条件均值一致

$$\begin{equation}
\begin{array}{rcl}
E[t\vert \mathbb{x}]=\int{tp(t\vert \mathbb{x})}dt=y(\mathbb{x}),\mathbb{w}))
\end{array}
\end{equation}$$

其中 $p(t\vert \mathbb{x})=p(t\vert \mathbb{x},\mathbb{w},\beta)$ 。需要注意的是高斯噪声假设隐含t在给定x的条件分布是单峰的，这个性质可能对于某些应用不太合适。作为扩展，我们可以采用混合高斯分布。

$\boldsymbol{X}=\{\mathbb{x}_1,\dots,\mathbb{x}_N\}$ ，其对应的值为 $\mathbb{t}=\{t_1,\dots,t_N\}$ ，那么

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{t}\vert \boldsymbol{X},\mathbb{w},\beta)=\prod_{n=1}^N\mathcal{N}(t\vert y(\mathbb{x}_n,\mathbb{w}),\beta^{-1})
\end{array}
\end{equation}$$

为了使公式保持整齐，我们可以将上式写成

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{t}\vert \mathbb{w},\beta) &=& \prod_{n=1}^N\mathcal{N}(t\vert \mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}_n),\beta^{-1}) \\
\ln p(\mathbb{t}\vert \mathbb{w},\beta) &=& \frac{N}{2}\ln\beta-\frac{N}{2}\ln2\pi-\beta E_D(\mathbb{w}) \\
E_D(\mathbb{w}) &=& \frac{1}{2}\sum_{n=1}^N\left\{t_n-\mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}_n)\right\}^2
\end{array}
\end{equation}$$

要使 $p(\mathbb{t}\vert \mathbb{w},\beta)$ 最大，那么

$$\begin{equation}
\begin{array}{rcl}
0 &=& \frac{\partial{\ln p(\mathbb{t}\vert \mathbb{w},\beta)}}{\partial{\mathbb{w}}}  \\
0 &=& \sum_{n=1}^N\left\{t_n-\mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}_n)\right\}\boldsymbol{\phi}(\mathbb{x}_n)^T \\
\Rightarrow \mathbb{w}_{ML} &=& (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbb{t} \\
\\
\boldsymbol{\Phi} &=& \begin{bmatrix}
\phi_0(\mathbb{x}_1) & \cdots & \phi_{M-1}(\mathbb{x}_1) \\
\phi_0(\mathbb{x}_2) & \cdots & \phi_{M-1}(\mathbb{x}_2) \\
\vdots & \ddots & \vdots \\
\phi_0(\mathbb{x}_N) & \cdots & \phi_{M-1}(\mathbb{x}_N) \\
\end{bmatrix}
\end{array}
\end{equation}$$

如果我们将偏置参数 $w_0$ 提出来，那么

$$\begin{equation}
\begin{array}{rcl}
E_D(\mathbb{w}) &=& \frac{1}{2}\sum_{n=1}^N\left\{t_n-w_0-\sum_{j=1}^{M-1}w_j\phi_j(\mathbb{x}_n)\right\}^2 \\
\Rightarrow w_0 &=& \frac{1}{N}\sum_{n=1}^Nt_n-\sum_{j=1}^{M-1}w_j\left\{\frac{1}{N}\sum_{n=1}^N\phi_j(\mathbb{x}_n)\right\}
\end{array}
\end{equation}$$

由上面公式我们可以看出，偏置参数 $w_0$ 补偿平均目标值与基函数加权平均值的差异。

$$\begin{equation}
\begin{array}{rcl}
\frac{1}{\beta_{ML}} &=& \frac{1}{N}\sum_{n=1}^N\left\{t_n-\mathbb{w}_{ML}^T\boldsymbol{\phi}(\mathbb{x}_n)\right\}^2 \\
\end{array}
\end{equation}$$

我们可以得到预测值与噪声的精度无关，但噪声的精度可以作为衡量预测值与目标值差异的一个标准。

### 3.1.2 最小二乘的几何形状

首先考虑坐标轴为 $t_n$ 的N维空间，那么 $\mathbb{t}_n=\{t_1,\dots,t_N\}^T$ 就是这个空间里的一个向量。那么 $\boldsymbol{\varphi}_j=\{\phi_j(\mathbb{x}_1),\dots, \phi_j(\mathbb{x}_N)\}^T$ 也是N为空间里的向量。$\boldsymbol{\mathbb{y}}=\{y(\mathbb{x}_1, \mathbb{w}),\dots,y(\mathbb{x}_N, \mathbb{w})\}^T$

如果基函数的数量 $M$ 小于训练数据集的数量 $N$，那么我们可以找到一个M维的子空间 $\mathcal{S}$ 来表示 $\boldsymbol{\varphi}_j$ ，而 $\boldsymbol{\mathbb{y}}$ 是 $\boldsymbol{\varphi}_j$ 向量的任意线性组合，所以 $\boldsymbol{\mathbb{y}}$ 可以落在M维子空间 $\mathcal{S}$ 的任意位置上。 所以最小二乘法的解就变成了 $\mathbb{t}$ 在子空间 $\mathcal{S}$ 的投影。

![geometry_of_least_squares](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/geometry_of_least_squares.png?raw=true)

### 3.1.3 顺序学习

最小二乘法的顺序学习可以采用随机梯度下降法

$$\begin{equation}
\begin{array}{rcl}
\mathbb{w}^{\tau+1} &=& \mathbb{w}^{\tau}-\eta\nabla E_n \\
\mathbb{w}^{\tau+1} &=& \mathbb{w}^{\tau}+\eta(t_n-\mathbb{w}^T\phi(\mathbb{x}_n))\phi(\mathbb{x}_n) \\
\end{array}
\end{equation}$$

### 3.1.4 正则化的最小二乘法

在第一章中，我们介绍了一种正则化的方法来控制过拟合，即

$$\begin{equation}
\begin{array}{rcl}
E(\mathbb{w}) &=& E_D(\mathbb{w}) + \lambda E_W(\mathbb{w})  \\
E_D(\mathbb{w})  &=& \frac{1}{2}\sum_{n=1}^N\left\{t_n-\mathbb{w}^T\boldsymbol{\phi}(x_n)  \right\}^2 \\
E_W(\mathbb{w}) &=& \frac{1}{2}\mathbb{w}^T\mathbb{w}
\end{array}
\end{equation}$$

上面的 $E_W$ 是比较常见的权值衰减，也可以采用其他的正则化方法。权值衰减的一个优点是保持整个损失函数仍然是权值的二项式。正则化方法会使权值收缩，从而减小噪声的影响。

$$\begin{equation}
\begin{array}{rcl}
\frac{1}{2}\sum_{n=1}^N\left\{t_n-\mathbb{w}^T\boldsymbol{\phi}(x_n)  \right\}^2 + \frac{\lambda}{2}\sum_{j=1}^{M-1}\vert w_j\vert^q
\end{array}
\end{equation}$$

![regularization](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/regularization.png?raw=true)

对于上式我们可以将其看成约束条件为 $\sum_{j=1}^{M-1}\vert w_j\vert^q\le \eta$ 的拉格朗日乘子，那么

![lasso_vs_decay](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/lasso_vs_decay.png?raw=true)

由于上图中的损失函数是 $w_1,w_2$ 的二项式函数，所以为圆形。当 $\lambda$ 增加时，会驱使更多的权值趋向0。

### 3.1.5 多个输出值

当有 K 个输出值时，那么权值 $W$ 为MxK的矩阵。之后的分析是一致的。

## 3.2 Bias-Variance

我们知道如果训练数据集过小，而模型太复杂的话，会导致过拟合现象，但是如果为了减小过拟合现象而限制模型的灵活性，又可能会捕捉不到数据的重要性质。因此我们引入了正则化项，但是如何决定 $\lambda$ 仍然是个难题。

第一章中我们得到了 $E(L)$ 的表达式，

$$\begin{equation}
\begin{array}{rcl}
E(L) &=& \int{\{y(\mathbb{x})-h(\mathbb{x})\}^2p(\mathbb{x})}\mathrm{d}\mathbb{x}+\int{\{t-h(\mathbb{x})\}^2p(x,t)}\mathrm{d}x\mathrm{d}t \\
h(\mathbb{x}) &=& \int{tp(t\vert \mathbb{x})}dt=E[t\vert \mathbb{x}]
\end{array}
\end{equation}$$

上式的 $h(\mathbb{x})$ 是最佳预测，而 $y(\mathbb{x})$ 是实际预测。其中第二项是固有的噪声，第一项跟我们选取的 $y(\mathbb{x})$ 相关，当我们有充足的训练数据的时候 $y(\mathbb{x})=h(\mathbb{x})$ ，但是往往训练数据集是有限的。

如果我们用 $y(\mathbb{x},\mathbb{w})$ 来对 $h(\mathbb{x})$ 建模，在贝叶斯方法中，这个模型的不确定性由 $p(\mathbb{w}\vert \mathbb{x})$ 决定；而对于频率方法来说，需要基于一个特定数据集 $\mathcal{D}$ 来对 $\mathbb{w}$ 进行估计。对于任意数据集 $\mathcal{D}$ ，我们能够得到 $y(\mathbb{x};\mathcal{D})$ ，那么评测一个学习算法的性能时采用对所有数据集的集成平均。

假设我们有一个数据集 $\mathcal{D}$ ，则第一项的被积函数可以写成

$$\begin{equation}
\begin{array}{rcl}
\{y(\mathbb{x};\mathcal{D})-h(\mathbb{x})\}^2
\end{array}
\end{equation}$$

实际预测值 $y(\mathbb{x};\mathcal{D})$ 相对于 $\mathcal{D}$ 的平均值为 $E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]$ ，那么上式可以写成

$$\begin{equation}
\begin{array}{rcl}
&&\{y(\mathbb{x};\mathcal{D})-E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]+E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]-h(\mathbb{x})\}^2 \\
&=& \left\{y(\mathbb{x};\mathcal{D})-E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]\right\}^2+ \{E_\mathcal{D}[y(\mathbb{x};\mathcal{D})] -h(\mathbb{x}) \}^2\\
&+&2\{y(\mathbb{x};\mathcal{D})-E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]\}\{E_\mathcal{D}[y(\mathbb{x};\mathcal{D})] -h(\mathbb{x}) \} \\
\\
E_D[\{y(\mathbb{x};\mathcal{D}) -h(\mathbb{x}) \}^2] &=& \begin{matrix}\underbrace{\{E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]-h(\mathbb{x})\}^2}\\\text{(bias)}^2\end{matrix} +\begin{matrix}\underbrace{E_\mathcal{D}[\{y(\mathbb{x};\mathcal{D})-E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]\}^2]}\\\text{variace}\end{matrix}
\end{array}
\end{equation}$$

$h(\mathbb{x})$ 是最佳预测不依赖于 $\mathcal{D}$。偏置(bias)项表示平均值与最佳预测值的差异，方差(variance)表示了个体数据与平均值的震荡程度，即 $y(\mathbb{x};\mathcal{D})$ 对于特殊的数据集敏感度。因此期望平方差可以写成

$$\begin{equation}
\begin{array}{rcl}
\text{expected loss} &=& \text{bias}^2+\text{variance}+\text{noise}\\
\text{bias}^2 &=& \int\{E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]-h(\mathbb{x})\}^2p(\mathbb{x})d\mathbb{x} \\
\text{variance} &=& \int E_\mathcal{D}[\{y(\mathbb{x};\mathcal{D})-E_\mathcal{D}[y(\mathbb{x};\mathcal{D})]\}^2] p(\mathbb{x})d\mathbb{x} \\
\text{noise} &=& \int\{h(\mathbb{x}) - t\}^2p(\mathbb{x},t)d\mathbb{x}dt
\end{array}
\end{equation}$$

由于噪声是固有的，因此我们只能优化bias和variance，对于灵活(flexible)模型有较小的偏置极大方差，对于相对刚性(rigid)的模型具有较大的偏置和较小的方差。
$$\begin{equation}
\begin{array}{rcl}
p(t\vert \mathbb{x},\mathbb{w},\beta)=\mathcal{N}(t\vert y(\mathbb{x},\mathbb{w}),\beta)
\end{array}
\end{equation}$$

![bias_variance_lambda](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/bias_vs_variance_lambda.png?raw=true)

这个图在方差方面不太直观，可以参考第一章图1.4.

![bias_variance](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/bias_variance.png?raw=true)

尽管从频率观点上，bias-variace分解能够提供一些有趣(interesting)的见解，但是它是基于对集成数据集上的，而在实际中我们只有一个数据集。如果我们有一系列数据集，合并成一个数据集也可以减小过拟合现象。

## 3.3 贝叶斯线性回归

贝叶斯方法能够避免最大似然方法导致的过拟合现象，并且在只使用训练数据的情况下自动决定模型的复杂性。

### 3.3.1 参数分布

[第二章](chapter2_prml.md)中我们介绍了在贝叶斯方法中最重要的是共轭先验，在线性回归中我们知道似然函数为 $p(\mathbb{t}\vert \mathbb{w},\beta)=\prod_{n=1}^N\mathcal{N}(t\vert \mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}_n),\beta^{-1})$ 是关于 $\mathbb{w}$ 二项式的指数函数，那么共轭先验是

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{w}) &=& \mathcal{N}(\mathbb{w} \vert \mathbb{m}_0,\boldsymbol{S}_0) \\
p(\mathbb{w}\vert \mathbb{t}) &=& \mathcal{N}(\mathbb{w} \vert \mathbb{m}_N,\boldsymbol{S}_N) \\
\mathbb{m}_N &=& \boldsymbol{S}_N(\boldsymbol{S}_0^{-1}\mathbb{m}_0+\beta\boldsymbol\Phi^T\mathbb{t}) \\
\boldsymbol{S}_N^{-1} &=& \boldsymbol{S}_0^{-1}+\beta\boldsymbol\Phi^T\boldsymbol\Phi
\end{array}
\end{equation}$$

上述后验概率的推导可以利用 completing the square方法得出。由于后验概率是高斯分布，因此它的模(mode)即均值(mean)，所以 $\mathbb{w}_{MAP}=\mathbb{m}_N$ 。如果 $\boldsymbol{S}_0=\alpha^{-1} \boldsymbol{I}$ ，当 $\alpha \to 0$ 时，先验概率就表示对整个空间上的参数没有偏向，那么 $\mathbb{m}_N$ 就变成了 $\mathbb{w}_{ML}$ 。

考虑一个特殊共轭先验 $p(\mathbb{w}\vert \alpha)=\mathcal{N}(\mathbb{w}\vert \boldsymbol{0},\alpha^{-1}\boldsymbol{I})$ ，那么

$$\begin{equation}
\begin{array}{rcl}
\mathbb{m}_N &=& \beta\boldsymbol{S}_N\boldsymbol\Phi^T\mathbb{t} \\
\boldsymbol{S}_N^{-1} &=& \alpha\boldsymbol{I}+\beta\boldsymbol\Phi^T\boldsymbol\Phi \\
\Rightarrow \ln p(\mathbb{w}\vert \mathbb{t}) &=& -\frac{\beta}{2}\sum_{n=1}^N\{t_n-\mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}_n) \}^2 -\frac{\alpha}{2}\mathbb{w}^T\mathbb{w}+\text{const}
\end{array}
\end{equation}$$

从 $\ln p$ 可以看到权值衰减的平方差损失函数符合贝叶斯思想。

![beyasian learning](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/bayesian_learning.png?raw=true)


### 3.3.2 预测分布

但是需要注意到是到目前为止，还是进行了点估计，上面的方法又叫做model selection，即选择后验概率最大的模型；下面我们介绍model averaging，即将多个模型加权平均从而得到预测分布。给定一个新的输入 $\mathbb{x}$ 预测它的值 $t$ ，

$$\begin{equation}
\begin{array}{rcl}
p(t\vert \mathbb{t},\alpha,\beta) &=& \int p(t\vert \mathbb{w},\beta)p(\mathbb{w}\vert \mathbb{t},\alpha,\beta) d\mathbb{w} \\
p(t\vert \mathbb{w},\beta) &=& \mathcal{N}(t\vert \mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}),\beta^{-1}) \\
p(\mathbb{w} \vert \mathbb{t},\alpha,\beta) &=& \mathcal{N}(\mathbb{w}\vert \mathbb{m}_N, \boldsymbol{S}_N) \\
\end{array}
\end{equation}$$

上面的公式为了简化概念，省略了 $\mathbb{t}$ 相对应的输入值

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{x}) &=& \mathcal{N}(\mathbb{x} \vert \mu, \Lambda^{-1}) \\
p(\mathbb{y} \vert \mathbb{x}) &=& \mathcal{N}(\mathbb{y} \vert \boldsymbol{A}\mathbb{x}+b, L^{-1}) \\
\Rightarrow p(\mathbb{y}) &=& \mathcal{N}(\mathbb{y} \vert \boldsymbol{A}\mu+b, L^{-1}+\boldsymbol{A}\Lambda^{-1}\boldsymbol{A}^T) \\
\Rightarrow p(t\vert \mathbb{x},\mathbb{t},\alpha, \beta) &=& \mathcal{N}(t\vert \mathbb{m}_N^T\boldsymbol{\phi}(\mathbb{x}),\sigma^2_N(\mathbb{x})) \\
\sigma^2_N(\mathbb{x}) &=& \frac{1}{\beta} + \boldsymbol{\phi}(\mathbb{x})^T\boldsymbol{S}_N\boldsymbol{\phi}(\mathbb{x})
\end{array}
\end{equation}$$

$\sigma^2_N(\mathbb{x})$ 的第二项表示相对于参数 $\mathbb{w}$ 的不确定性，当 $N\to\infty$ 时，第二项趋向于0。

![predictive distribution](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/bayesian_prediction.png?raw=true)

从上图我们可以看出预测的不确定性依赖于 $x$，当 $x$ 在训练数据邻近时，不确定性大大降低，而且当训练数据增多时，整体不确定性降低。

下图是从后验概率抽取参数，并且画出它们的曲线

![bayesian curve sample](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/bayesian_weight_samples.png?raw=true)

### 3.3.3 等价核(Equivalent Kernel)

在model selection方法中，加入我们将均值代入到 $y(\mathbb{x},\mathbb{w})=\mathbb{w}^T\boldsymbol{\phi}(\mathbb{x})$ ，得到

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x},\mathbb{m}_N)=\mathbb{m}^T\boldsymbol{\phi}(\mathbb{x})=\beta\boldsymbol{\phi}(\mathbb{x})\boldsymbol{S}_N\boldsymbol{\Phi}^T\mathbb{t}=\sum_{n=1}^N\beta\boldsymbol{\phi}(\mathbb{x})^T\boldsymbol{S}_N\boldsymbol{\phi}(\mathbb{x}_n)t_n \\
\end{array}
\end{equation}$$

进一步写成

$$\begin{equation}
\begin{array}{rcl}
y(\mathbb{x},\mathbb{m}_N) &=& \sum_{n=1}^Nk(\mathbb{x},\mathbb{x}_n)t_n \\
k(\mathbb{x},\mathbb{x}') &=& \beta\boldsymbol{\phi}(\mathbb{x})^T\boldsymbol{S}_N\boldsymbol{\phi}(\mathbb{x}')
\end{array}
\end{equation}$$

上式 $k$ 即equivalent kernel。我们可以将 $k(\mathbb{x},\mathbb{x}_n)$ 看成 $t_n$ 相对应的权值。

![equivalent kernel](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/equivalent_kernel.png?raw=true)

weight local evidence more strongly than distant evidence。

$$\begin{equation}
\begin{array}{rcl}
\text{cov}[y(\mathbb{x}),y(\mathbb{x}')] &=& \text{cov}[\boldsymbol{\phi}(\mathbb{x})^T\mathbb{w},\mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}')] \\
&=& \boldsymbol{\phi}(\mathbb{x})^T\boldsymbol{S}_N\boldsymbol{\phi}(\mathbb{x}') =\beta^{-1}k(\mathbb{x},\mathbb{x}')
\end{array}
\end{equation}$$

由此我们知道在预测值附近的点高度相关，而对于两个距离比较远的点相关性比较小。

## 3.4 贝叶斯模型比较

最大似然方法导致的过拟合现象能够通过边缘化(累加或积分)模型的参数而不是对参数进行点估计来避免。假设我们想比较L个模型 $\{\mathcal{M}_i\},i=1,\dots,L$ 。模型表示对数据集 $\mathcal{D}$ 的概率分布。$p(\mathcal{M}_i)$ 表示刚开始我们对这个分布的不确定性，即先验，那么给定一个数据集 $\mathcal{D}$ 后，

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{M}_i \vert \mathcal{D}) \propto p(\mathcal{D} \vert \mathcal{M}_i)p(\mathcal{M}_i)
\end{array}
\end{equation}$$

$p(\mathcal{D} \vert \mathcal{M}_i)$ 是模型的证据(model evidence)，即表示模型被当前数据集的偏向性(preference)，又叫边缘似然(marginal likelihood)。贝叶斯因子： $p(\mathcal{D} \vert \mathcal{M}_i)/p(\mathcal{D} \vert \mathcal{M}_j)$

对于model averaging来说

$$\begin{equation}
\begin{array}{rcl}
p(t\vert \mathbb{x},\mathcal{D}) = \sum_{i=1}^Lp(t\vert \mathbb{x}, \mathcal{M}_i,\mathcal{D})p(\mathcal{M}_i \vert \mathcal{D})
\end{array}
\end{equation}$$

很明显对于model averaging来说，计算量是一个比较大的限制，因此有时候我们可以用最大后验概率的模型来近似model averaging，这叫做model selection。模型由参数 $\mathbb{w}$ 控制，那么

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{D} \vert \mathcal{M}_i) = \int p(\mathcal{D} \vert \mathbb{w}, \mathcal{M}_i) p(\mathbb{w} \vert \mathcal{M}_i) d\mathbb{w}
\end{array}
\end{equation}$$

$\mathcal{M}_i$ 相当于模型的超参数(hyper-parameter)，比如线性回归中基函数与基函数的个数；而 $\mathbb{w}$ 则是这个模型学到的参数，很明显在相同的超参数下，参数 $\mathbb{w}$ 有多重可能性。

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{w} \vert \mathcal{D} , \mathcal{M}_i) =\frac{p(\mathcal{D} \vert \mathbb{w}, \mathcal{M}_i) p(\mathbb{w} \vert \mathcal{M}_i) }{p(\mathcal{D} \vert  \mathcal{M}_i)}
\end{array}
\end{equation}$$

下面我们考虑在模型 $\mathcal{M}_i$ 情况下，并且只有一个参数 $w$ ，后验概率 $p(w \vert \mathcal{D}) \propto p(\mathcal{D} \vert w ) p(w)$ ，假设后验概率分布集中分布在 $w_{MAP}$ ，宽度为 $\Delta w_{posterior}$，假如我们进一步假设先验为 $p(w)=1/\Delta w_{prior}$

![Bayesian model assumption](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/bayesian_model_comparison.png?raw=true)

$$\begin{equation}
\begin{array}{rcl}
p(\mathcal{D}) &=& \int p(\mathcal{D}\vert w)p(w) dw \simeq p(\mathcal{D}\vert w_{MAP})\frac{\Delta w_{posterior}}{\Delta w_{prior}} \\
\Rightarrow \ln p(\mathcal{D}) &\simeq& \ln p(\mathcal{D}\vert w_{MAP})+\ln\left(\frac{\Delta w_{posterior}}{\Delta w_{prior}}\right)
\end{array}
\end{equation}$$

第一项表示模型在最大后验对数据的拟合程度；第二项是惩罚项，因为 $\Delta w_{posterior} < \Delta w_{prior}$，所以这项是负值，并且当模型越复杂时，$\Delta w_{posterior}$ 就越小，从而使惩罚越重(对于相同的模型 $\mathcal{M}_i$，惩罚项相同？)。

假设有M个参数，并且所有的参数有相同的比率 $\Delta w_{posterior} / \Delta w_{prior}$ ，那么

$$\begin{equation}
\begin{array}{rcl}
\Rightarrow \ln p(\mathcal{D}) &\simeq& \ln p(\mathcal{D}\vert \mathbb{w}_{MAP})+M\ln\left(\frac{\Delta w_{posterior}}{\Delta w_{prior}}\right)
\end{array}
\end{equation}$$

此时模型的复杂性由M控制，当M增加时，第一项增加，当模型越复杂，它能更好的拟合数据，从而 $p(\mathcal{D}\vert \mathbb{w}_{MAP})$ 增加，故第一项增加；第二项减小，这是因为 $\ln\left(\frac{\Delta w_{posterior}}{\Delta w_{prior}}\right)$ 为负，故M增加，第二项减小，从而最优解是这两项的平衡点(trade-off)。

![models comparison which have different complexity](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/three_models_comparison.png?raw=true)

由上图我们知道，采用的模型跟数据量的大小有关。

贝叶斯模型能够只用训练数据选择最佳模型(理论上？)，假设我们有两个模型 $\mathcal{M}_1,\mathcal{M}_2$ ，而真实模型为 $\mathcal{M}_1$，那么

$$\begin{equation}
\begin{array}{rcl}
\int p(\mathcal{D} \vert \mathcal{M}_1) \ln\frac{p(\mathcal{D} \vert \mathcal{M}_1)}{p(\mathcal{D} \vert \mathcal{M}_2)} d\mathcal{D} = -\int p(\mathcal{D} \vert \mathcal{M}_1) \ln\frac{p(\mathcal{D} \vert \mathcal{M}_2)}{p(\mathcal{D} \vert \mathcal{M}_1)} d\mathcal{D}
\end{array}
\end{equation}$$

上式为相对熵，表示 $\mathcal{M}_1,\mathcal{M}_2$ 的不相似性，明显理论上是可以单单采用训练数据就可以选择正确模型，但是在有多个模型的时候，计算量十分复杂。

## 3.5 The Evidence Approximation

Fully Bayesian中要求边缘化hyper-parameter，但是这往往是analytically intractable的，因此我们用逼近思想去分析，

$$\begin{equation}
\begin{array}{rcl}
p(t\vert \mathbb{t}) = \int\int\int p(t\vert \mathbb{w}, \beta) p(\mathbb{w} \vert \mathbb{t}, \alpha, \beta) p(\alpha,\beta\vert \mathbb{t}) d\mathbb{w}d\alpha d\beta \\
\end{array}
\end{equation}$$

如果后验分布 $p(\alpha,\beta\vert \mathbb{t})$ 集中分布在 $\hat{\alpha},\hat{\beta}$ 上，那么上式近似于

$$\begin{equation}
\begin{array}{rcl}
p(t\vert \mathbb{t}) \simeq p(t\vert \mathbb{t}, \hat{\alpha},\hat{\beta}) = \int p(t\vert \mathbb{w}, \hat{\beta}) p(\mathbb{w} \vert \mathbb{t}, \hat{\alpha},\hat{\beta}) d\mathbb{w} \\
\end{array}
\end{equation}$$

所以现在我们的目标最变成了如何确定 $\hat{\alpha},\hat{\beta}$ ，由于 $p(\alpha,\beta\vert \mathbb{t})\propto p(\mathbb{t}\vert \alpha,\beta )p(\alpha,\beta)$，如果我们对 $p(\alpha,\beta)$ 没有偏向，那么变成了 $p(\alpha,\beta\vert \mathbb{t})\propto p(\mathbb{t}\vert \alpha,\beta )$，从而目标变成了求 $p(\mathbb{t}\vert \alpha,\beta )$

### 3.5.1 Evalution of Evidence Function

$$\begin{equation}
\begin{array}{rcl}
p(\mathbb{t}\vert \alpha,\beta) &=& \int p(\mathbb{t}\vert \mathbb{w},\beta)p(\mathbb{w} \vert \alpha) d\mathbb{w} \\
p(\mathbb{t}\vert \mathbb{w},\beta) &=& \sum_{n=1}^N\mathcal{N}(t_n\vert \mathbb{w}^T\boldsymbol{\phi}(\mathbb{x}_n),\beta^{-1}) \\
p(\mathbb{w}\vert \alpha) &=& \mathcal{N}(\mathbb{w}\vert \boldsymbol{0}, \alpha^{-1}\boldsymbol{I}) \\
\Rightarrow p(\mathbb{t}\vert \alpha,\beta) &=& \left(\frac{\beta}{2\pi} \right)^{N/2} \left(\frac{\alpha}{2\pi} \right)^{M/2}\int \exp\{-E(\mathbb{w})\} d\mathbb{w} \\
E(\mathbb{w}) &=& \beta E_D(\mathbb{w})+\alpha E_W(\mathbb{w}) = \frac{\beta}{2}\|\mathbb{t}-\boldsymbol{\Phi}\mathbb{w}\|^2+\frac{\alpha}{2}\mathbb{w}^T\mathbb{w}
\end{array}
\end{equation}$$

我们想得到高斯函数的指数形式即 $E(\mathbb{w})=\frac{1}{2}(\mathbb{w}-\mathbb{m}_N)^T\boldsymbol{A}(\mathbb{w}-\mathbb{m}_N)+E(\mathbb{m}_N)$，第二项是与 $\mathbb{w}$ 无关的项，根据completing the square方法，我们得到

$$\begin{equation}
\begin{array}{rcl}
\boldsymbol{A} &=& \alpha\boldsymbol{I}+\beta \boldsymbol{\Phi}^T\boldsymbol{\Phi} \\
\mathbb{m}_N &=& \beta \boldsymbol{A}^{-1}\boldsymbol{\Phi}^T\mathbb{t} \\
E(\mathbb{m}_N) &=& \frac{\beta}{2}\|\mathbb{t}-\boldsymbol{\Phi}\mathbb{m}_N\|^2+\frac{\alpha}{2}\mathbb{m}_N^T\mathbb{m}_N
\end{array}
\end{equation}$$

所以

$$\begin{equation}
\begin{array}{rcl}
\int \exp\{-E(\mathbb{w})\} d\mathbb{w} &=& \exp\{-E(\mathbb{m}_N)\}\int \exp\left\{-\frac{1}{2}(\mathbb{w}-\mathbb{m}_N)^T\boldsymbol{A}(\mathbb{w}-\mathbb{m}_N) \right\} d\mathbb{w} \\
&=& \exp\{-E(\mathbb{m}_N) \}(2\pi)^{M/2}\vert \boldsymbol{A} \vert^{-1/2} \\
\Rightarrow \ln  p(\mathbb{t}\vert \alpha,\beta) &=& \frac{M}{2}\ln\alpha+\frac{N}{2}\ln\beta - E(\mathbb{m}_N)-\frac{1}{2}\ln\boldsymbol{\vert A\vert}-\frac{N}{2}\ln 2\pi
\end{array}
\end{equation}$$

下图是第一章中曲线拟合的model evidence

![evidence function](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/evidence_function_plot.png?raw=true)

### 3.5.2 Maximizing the evidence function

首先我们考虑 $p(\mathbb{t}\vert \alpha,\beta)$ 相对于 $\alpha$ 的偏导，在这之前我们定义 $(\beta \boldsymbol{\Phi}^T\boldsymbol{\Phi})\mathbb{u}_i =\lambda_i\mathbb{u}_i$，那么 $A$ 的特征值即为 $\alpha+\lambda_i$。

$$\begin{equation}
\begin{array}{rcl}
\frac{d}{d\alpha}\ln\boldsymbol{\vert A\vert} &=& \frac{d}{d\alpha}\prod_{i}\ln(\lambda_i+\alpha)=\sum_i\frac{1}{\lambda_i+\alpha} \\
\Rightarrow 0 &=& \frac{M}{2\alpha}-\frac{1}{2}\mathbb{m}_N^T\mathbb{m}_N-\frac{1}{2}\sum_i\frac{1}{\lambda_i+\alpha} \\
\Rightarrow \alpha\mathbb{m}_N^T\mathbb{m}_N &=& M-\alpha\sum_i\frac{1}{\lambda_i+\alpha} = \gamma \\
\Rightarrow \gamma &=& \sum_i\frac{\lambda_i}{\lambda_i+\alpha} \\
\Rightarrow \alpha &=& \frac{\gamma}{\mathbb{m}_N^T\mathbb{m}_N}
\end{array}
\end{equation}$$

需要注意的是 $\alpha = \frac{\gamma}{\mathbb{m}_N^T\mathbb{m}_N}$ 并不是 $\alpha$ 的直接解，因为 $\gamma,\mathbb{m}_N$ 都依赖于 $\alpha$。我们采用迭代过程来对 $\alpha$ 进行求解，首先任意初始化 $\alpha$，然后求 $\mathbb{m}_N=\beta\boldsymbol{S}_N\boldsymbol{\Phi}^T\mathbb{t}$，再求 $\gamma$，递归求解 $\alpha$

同理求 $\beta$ 的偏导

$$\begin{equation}
\begin{array}{rcl}
\frac{d}{d\beta}\ln\boldsymbol{\vert A\vert} &=& \frac{d}{d\beta}\prod_{i}\ln(\lambda_i+\alpha)=\frac{1}{\beta}\sum_i\frac{\lambda_i}{\lambda_i+\alpha}=\frac{\gamma}{\beta} \\
\Rightarrow \frac{1}{\beta} &=& \frac{1}{N-\gamma}\sum_{n=1}^N\{\mathbb{t}_n-\mathbb{m}_N^T\boldsymbol{\phi}(\mathbb{x}_n)\}^2
\end{array}
\end{equation}$$

明显也要用迭代过程来对 $\beta$ 进行求解。而且我们知道 $\alpha,\beta$ 的值依赖于训练数据。

### 3.5.3 有效参数的数量

![parameters measurement](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/chap3/interpretation_alpha.png?raw=true)

当 $\alpha\to0$ 时，此时的模是似然函数的解，而 $A=\beta \boldsymbol{\Phi}^T\boldsymbol{\Phi}$，也就是说 $\lambda_i$ 是一个参数被数据决定的程度( $\lambda_i$ measures how strongly one parameter is determined by the data)。那么 $\alpha$ 就是所有参数被先验决定的程度。 $\mathbb{w}_{MAP_i} = \frac{\lambda_i}{\lambda_i+\alpha}\mathbb{w}_{ML_i}$，如果 $\lambda_i\ll\alpha$，那么 $w_i\to0$ ，即数据对这个参数不敏感，由上面分析可以知道 $\gamma_i=\frac{\lambda_i}{\lambda_i+\alpha}$ 能够估量有效参数的数量。

现在我们考虑 $\beta$，在贝叶斯分析中 $\frac{1}{\beta_{MAP}} = \frac{1}{N-\gamma}\sum_{n=1}^N\{\mathbb{t}_n-\mathbb{m}_N^T\boldsymbol{\phi}(\mathbb{x}_n)\}^2$，而在最大似然估计中 $\frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^N\{\mathbb{t}_n-\mathbb{w}_{ML}^T\boldsymbol{\phi}(\mathbb{x}_n)\}^2$，因为在贝叶斯分析中的参数有效数量取决于 $\gamma$，因此要补偿最大似然估计从而使其无偏。

当 $N\gg M$ 时，$r=M$，此时

$$\begin{equation}
\begin{array}{rcl}
\beta &=& \frac{N}{2E_D(\mathbb{m}_N)} =  \frac{N}{\sum_{n=1}^N\{\mathbb{t}_n-\mathbb{w}_{ML}^T\boldsymbol{\phi}(\mathbb{x}_n)\}^2}\\
\alpha &=& \frac{M}{2E_W(\mathbb{m}_N)} = \frac{M}{\mathbb{w}^T\mathbb{w}} \\
\end{array}
\end{equation}$$
