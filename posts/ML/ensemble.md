## 综述

常见的集成框架有Bagging 和 Boosting。但是经常看一些博客把Bootstrap与Bagging混淆，Bootstrap是一种统计方法，会用于评估一个estimator的性能(划分训练集与测试集)；而Bagging和Boosting则是集成框架，Bagging框架主要能减小variance，Boosting主要减小Bias(当然这两者都能减小variance/bias)。

## Bagging

输入：
1. 训练集 $D=\{(\mathbf{x_1}, \mathbf{y_1} ),(\mathbf{x_2}, \mathbf{y_2} ),\dots,(\mathbf{x_n}, \mathbf{y_n} ) \}$  
2. 基学习器算法： $\mathfrak{L}$
3. 学习器的个数: T

过程：
$$\begin{equation}
\begin{array}{lcl}
\text{for t = 1,2,...,T do} \\
\qquad h_t = \mathfrak{L}(D,D_{bs}) \\
\text{end for} \\
H(\mathbf{x}) = \arg\max_{y}\sum_{i=1}^T \mathbb{I}(h_t(\mathbf{x})=y)
\end{array}
\end{equation}$$

在Bagging中经常用 Bootstrap 来采样生成新的训练子集 $D_{bs}$，可能这就是经常混淆Bagging与Bootstrap的原因。

主要代表有随机森林。

## Boosting

与Bagging不同，Boosting先从出事训练集中训练出一个基学习器， 再根据基学习器的表现对训练样本分布进行调整，使得在先前基学习器分错的训练样本在后续的训练中得到更多的关注，而正确的样本关注度减小，训练完T个基学习器后再对他们进行加权结合。

主要代表有[AdaBoost](AdaBoost.md)。

## 总结

从以上两个框架的过程中，我们可以看到Bagging主要减小Variance，因为当采样越多越接近真实的样本分布；而Boosting则关注降低Bias，但是容易过拟合。
