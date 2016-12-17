## Why

### Traditional Mechine Learning

Traditional machine learning makes a basic assumption: the training and test data should be under the same distribution. That's a ideal situation.

### The goal of TrAdaBoost

TrAdaBoost allows users to utilize a small amount of newly labed data to leverage the old data to construct a high-quality classification mode for new data.

## How

### Some Defintion

1. diff-distribution training data whose distribution may differ from the test data. And our goal is to leverage this set of data to help construct a good classification mode. denote $X_d$
2. same-distribution training data which is under the same distribution of test data. This set of training data is very small. It helps vote on the usefulness of each of the diff-distribution instance. denote $X_s$
3. $Y=\{0,1\}$ is the set of category lables. $S={x_i^{t}}$ is test set. $T\subseteq \{X \times Y \}$
4. $l_i^t = |h_t(x_i)-c_t(x_i)|$ is the loss of the training instance $x_i$ suffered by the hepothesis $h_t$.

### Key Idea

Use boosting to filter out the diff-distribution training data that are very different from the same-distribution data by automatically adjusting the weights of training instances. The remaining diff-distribution data are treated as the additional training data which greatly boost the confidence of the learned model even when the same-distribution training data are scarce.

### Details

AdaBoost is applied to same-distribution training data to build the base of the model.

For diff-distribution training instances which are wrongly predicted due to distribution changes by the learned model, these instances could be those that are the most dissimilar to the same-distribution instances. **Add a mechanism to decrease the weights of these instances in order to weaken their impacts**.

if a diff-distribution training instance is mistakenly predicted, the instance may likely conflict with the same-distribution training data. Then, we decrease its training weight to reduce its effect through multiplying its weight by $\beta^{|h_t(x_i)−c(x_i)|}$. Note that $\beta^{|h_t(x_i)−c(x_i)|} \in (0, 1]$. Thus, in the next round, the misclassified diff-distribution training instances, which are dissimilar to the same-distribution ones, will affect the learning process less than the current round. After several iterations, the diff-distribution training instances that fit the same-distribution ones better will have larger training weights, while the diff-distribution training instances that are dissimilar to the samedistribution ones will have lower weights.

![TrAdaBoost_algorithm](https://raw.githubusercontent.com/lehyu/lehyu.github.com/master/image/TL/TrAdaBoost_algorithm.png)

## Theoretical Analysis

**Theorem 1** Covergence property

$$\begin{equation}
\begin{array}{rcl}
\frac{L_d}{N} \leq \min_{1 \leq i \leq n} \frac{L(x_i)}{N}+\sqrt{\frac{2lnn}{N}}+\frac{lnn}{N}
\end{array}
\end{equation}$$

It indicates that the average training loss $(L_d/N)$ through N iterations on the diff-distribution training data Td is not much (at most $\sqrt{2\ln {n}/N} + \ln{n}/N$) larger than the average training loss of the instance whose average training loss is minimum (that is, $min1≤i≤n L(xi)/N$ ).


**Theorem 2** analyzes the average weighted training loss on
diff-distribution data $T_d$.

$$\begin{equation}
\begin{array}{rcl}
\lim_{N \to \infty}\frac{\sum_{t=\lceil N/2 \rceil}^N\sum_{i=1}^n p_i^t l_i^t}{N-\lceil N/2 \rceil}
\end{array}
\end{equation}$$

This is why we vote from $\lceil N/2 \rceil$ to N.

**Theorem 3** shows the prediction error on the same-distribution data $T_s$.Let $I = \{i : h_f (x_i) \neq c(x_i)$ and $n + 1 ≤ i ≤ n + m\}$ . The prediction error on the same-distribution training data $T_s$ suffered by the final hypothesis $h_f$ is defined as  

$$\begin{equation}
\begin{array}{rcl}
\epsilon &=& Pr_{x\in T_s}[h_f(x)\neq c(x)]=|I|/m \\
&=& 2^{\lceil N/2 \rceil} \prod_{t=\lceil N/2 \rceil}^N\sqrt{\epsilon_t(1-\epsilon_t)}
\end{array}
\end{equation}$$

## Reference

[1 Boosting for Transfer Learning](http://delivery.acm.org/10.1145/1280000/1273521/p193-dai.pdf?ip=58.60.1.88&id=1273521&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E0871A888CCEFF346%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=688099428&CFTOKEN=60229314&__acm__=1477838973_ba5e16c8d030322bdd8c9b56cbc0c909)

[2 A desicion-theoretic generalization of on-line learning and an application to boosting](http://download.springer.com/static/pdf/96/chp%253A10.1007%252F3-540-59119-2_166.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fchapter%2F10.1007%2F3-540-59119-2_166&token2=exp=1477839965~acl=%2Fstatic%2Fpdf%2F96%2Fchp%25253A10.1007%25252F3-540-59119-2_166.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fchapter%252F10.1007%252F3-540-59119-2_166*~hmac=5b7ed6c6c3ef75f4fe17db92dd951b00c8a7a5af53dda4442a9ba59cc7945e0e)

[3 基于实例和特征的迁移学习算法研究](http://cdmd.cnki.com.cn/Article/CDMD-10248-2009140520.htm)
