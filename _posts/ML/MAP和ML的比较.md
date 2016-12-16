## 最大似然估计
设总体 $X$ 为离散型，其分布 $P(X=x)=p(x;\theta)$ 。设 $X_1,\dots,X_n$ 是来自 $X$ 的样本，则他们的联合分布律为：
$$\prod_{i=1}^{N}p(x_i;\theta)$$
那么 $\theta$ 的函数， $L(\theta)=\prod_{i=1}^{N}p(x_i;\theta)$ 为其似然函数。取 $\hat{\theta}$ 使 $L(x_1,\dots,x_n;\hat{\theta})=\max_{\theta}L(x_1,\dots,x_n;\theta)$，则 $\hat{\theta}$ 为参数 $\theta$ 的最大似然估计值。

求 $\hat\theta$ : $\frac{\partial L}{\partial \theta} = 0$

## 最大后验概率估计
$p(h|D)=\frac{p(D|h)p(h)}{p(D)}$，求
$$\hat{h}=argmax_{h \in H}p(h|D)$$
$$\hat{h}=argmax_{h \in H}(\log(p(D|h)) + \log(p(h)))$$
当 $p(h)$ 是一个与数据量无关的值时，当数据量$N \to \infty$ 时，MAP收敛于ML

## 参考
[最大似然估计-百度百科](http://baike.baidu.com/link?url=IJBEAJs9cwEYbJfOC0LlL8Spc1QnoGFGzdSAXvsRVWcBehvCA5qtDQ_y-VqVxt6OGP0tBQqTKpxYeJ8qJ2c3vK)
