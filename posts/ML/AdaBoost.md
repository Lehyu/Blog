## 概述

Boosting是一种将弱学习器提升为强学习器的算法，刚开始时用初始的数据训练一个基学习器，然后再根据先前基学习器的调整数据的分布，使得被错估的样本得到更多的关注(增加它们的权重)，减小正确的样本的关注度(降低其权重)，然后在训练新的学习器，直至基学习器的个数达到终止条件。那么算法最终的预测函数就为

$$\begin{equation}
\begin{array}{lcl}
H(\mathbf{x}) = \sum_i^T\alpha_ih_i(\mathbf{x})
\end{array}
\end{equation}$$

## 理论
Boosting里最出名的是AdaBoost， 我们下面将主要介绍AdaBoost。

对于二分类问题，AdaBoost的损失函数为

$$\begin{equation}
\begin{array}{lcl}
L = \mathbb{E}_{x\sim D}[e^{-f(x)H(x)}]
\end{array}
\end{equation}$$

要使 $H(x)$ 能令损失函数最小，那么对其求偏导为零，得

$$\begin{equation}
\begin{array}{lcl}
\frac{\partial{L}}{\partial{H(x)}} &= & -e^{-H(x)}P(f(x)=1|x)+e^{H(x)}P(f(x)=-1|x) \\
H(x) &=& \frac{1}{2}\ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x)} \\
sign(H(x)) &=& \begin{cases}\
1, & P(f(x)=1|x) > P(f(x)=-1|x) \cr
-1, & P(f(x)=1|x) < P(f(x)=-1|x)
\end{cases}
\end{array}
\end{equation}$$

由于 $H(x)$ 是加法模型因此我们可以采用向前分布的办法来求解问题(具体参考统计学习方法-李航，P144～P145). 迭代生成 $\alpha_t,h_t$ 。

$$\begin{equation}
\begin{array}{lcl}
L &=& \mathbb{E}_{x\sim D_t}[e^{-f(x)\alpha_th_t(x)}] \\
  &=&  \mathbb{E}_{x\sim D_t}[e^{-\alpha_t}\mathbb{I}(f(x)=h(x))+e^{\alpha_t}\mathbb{I}(f(x)\neq h(x))] \\
  &=& e^{-\alpha_t}P_{x\sim D_t}(f(x)=h(x)) + e^{\alpha_t}P_{x\sim D_t}(f(x)\neq h(x)) \\
  &=& e^{-\alpha_t}(1-\epsilon_t) + e^{\alpha_t}\epsilon_t \\

\Rightarrow \alpha_t &= & \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}
\end{array}
\end{equation}$$

现在我们已经得到 t 步基学习器 $h_t$ 及其权重 $\alpha_t$ ,对于下一个基学习器我们要根据已得到的基学习器更新数据的分布，希望下一个基学习器能够尽可能多的矫正前面基学习器错估的样本。直觉上我们要增加错估样本的权重而减小正确样本的权重，这样在下一次再错估这些样本就会提高下个基学习器的错误率从而减小其权重 $\alpha_{t+1}$

原始论文的更新方式
$$\begin{equation}
\begin{array}{lcl}
D_{t+1}(x) &=& D_t(x)\beta^{1-|h(x)-f(x)|} \\
\hat D_{t+1}(x_i) &=& \frac{D_{t+1}(x_i)}{\sum_{j=1}^ND_{t+1}(x_j)}
\end{array}
\end{equation}$$

由上面的公式我们知道错估的样本权重不变，降低正确样本的权重。另一种更新方式如下所示

$$\begin{equation}
\begin{array}{lcl}
\beta &=& \frac{\epsilon_t}{1-\epsilon_t} \\
D_{t+1}(x) &=& D_t(x)\times \begin{cases}\
\exp(-\alpha_t), & f(x)=h_t(x) \cr
\exp(\alpha_t), & otherwise
\end{cases} \\
\hat D_{t+1}(x_i) &=& \frac{D_{t+1}(x_i)}{\sum_{j=1}^ND_{t+1}(x_j)}
\end{array}
\end{equation}$$

## 伪代码
![AdaBoost](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/ensemble/adaboost.png?raw=true)

[python代码](https://github.com/Lehyu/pyml/blob/master/models/adaboost/adaboost.py)
