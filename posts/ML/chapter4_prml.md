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
