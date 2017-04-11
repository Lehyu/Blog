# Support Vector Machine

## SVM 最大间隔分类器

[线性分类器](chapter4_prml.md)中我们知道线性模型的一般形式为 $y(\mathbf{x})=\mathbf{w}^T\phi(\mathbf{x})+b$ ，对于二分类假设我们令 $t_n \in \{-1, 1\}$ ，那么对于 $t_n=1$ 的点有 $y(\mathbf{x}_n) > 0$，反之 $y(\mathbf{x}_n) < 0$ ， 所以我们有 $t_ny(\mathbf{x}) >0$ 。

那么任意一点 $\mathbf{x}$ 到分类平面的距离为 $\frac{t_ny(\mathbf{x}_n)}{\|\mathbf{w}\|}$ ， 我们知道在分类平面可能存在无数个，但是直观上看，当两个类别的点到分类平面的距离最大时，这个分类平面显然时最好的。

![hard_margin](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/svm/hard_margin.png?raw=true)

因此我们就希望优化一下目标函数

$$\begin{equation}
\begin{array}{rcl}
\arg\max_{\mathbf{w},b}\left\{ \frac{1}{\|\mathbf{w}\|}\min_n\left[ t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b )\right] \right\}
\end{array}
\end{equation}$$

对于上式，由于 $\mathbf{w}、b$ 可以缩放， 因此不失一般性的， 我们可以令 $t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b ) \ge 1$ ， 当 $t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b ) = 1$ 时， $(\mathbf{x}_n, t_n)$ 及为支持向量。所以优化目标函数可以转变为在约束条件 $t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b ) \ge 1$ 下，求

$$\begin{equation}
\begin{array}{rcl}
\arg\min_{\mathbf{w},b}\frac{1}{2}\|\mathbf{w}\|^2
\end{array}
\end{equation}$$

由拉格朗日乘子法得

$$\begin{equation}
\begin{array}{rcl}
L(\mathbf{w}, b, \mathbf{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{n=1}^N\alpha_n\left\{ t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b )- 1 \right\}
\end{array}
\end{equation}$$

对 $\mathbf{w}, b$ 对导得

$$\begin{equation}
\begin{array}{rcl}
\mathbf{w} &=& \sum_{n=1}^N\alpha_nt_n\phi(\mathbf{x}_n) \\
0 &=& \sum_{n=1}^N\alpha_nt_n
\end{array}
\end{equation}$$

从而得到最大间隔问题的对偶形式

$$\begin{equation}
\begin{array}{rcl}
L(\mathbf{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbf{x}_n, \mathbf{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0
\end{array}
\end{equation}$$

要是上述目标函数最优，必须满足 [KKT 条件](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)

$$\begin{equation}
\begin{array}{rcl}
\alpha_n &\ge& 0 \\
t_ny(\mathbf{x}_n)-1 &\ge& 0 \\
\alpha_n\{ t_ny(\mathbf{x}_n)-1  \} &=& 0
\end{array}
\end{equation}$$

## 软间隔

上面的推导中我们是基于 $t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b ) \ge 1$ ，及类别时线性可分的，当非线性可分时，我们引入松弛（slack）变量 $\xi_n \ge 0$ ，那么对于任意点 $\mathbf{x}_n$ 有 $t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b ) \ge 1-\xi_n$

![soft_margin](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/svm/soft_margin.png?raw=true)

那么此时的目标函数则是

$$\begin{equation}
\begin{array}{rcl}
&&\arg\min_{\mathbf{w},b}C\sum_{n=1}^N\xi_n + \frac{1}{2} \|\mathbf{w}\|^2 \\
\text{subject to}&& t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b ) &\ge& 1-\xi_n \\
&&\xi_n &\ge& 0
\end{array}
\end{equation}$$

由拉格朗日乘子法得

$$\begin{equation}
\begin{array}{rcl}
L(\mathbf{w}, b, \mathbf{\alpha}, \mathbf{\mu}) &=& C\sum_{n=1}^N\xi_n + \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{n=1}^N\mu_n\xi_n \\
&-&\sum_{n=1}^N\alpha_n\left\{t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b ) -1+\xi_n  \right\}  
\end{array}
\end{equation}$$

因此 KKT条件为

$$\begin{equation}
\begin{array}{rcl}
\alpha_n &\ge& 0\\
t_ny_n -1+\xi_n &\ge& 0\\
\alpha_n(t_ny_n -1+\xi_n) &=& 0\\
\mu_n &\ge& 0\\
\xi_n &\ge& 0\\
\mu_n\xi_n &=& 0\\
\end{array}
\end{equation}$$

对 $\mathbf{w}、b、\xi_n$ 求导得

$$\begin{equation}
\begin{array}{rcl}
\frac{\partial{L}}{\partial\mathbf{w}} = 0 &\Rightarrow& \mathbf{w} &=& \sum_{n=1}^N\alpha_nt_n\phi(\mathbf{x}_n) \\
\frac{\partial{L}}{\partial{b}} = 0 &\Rightarrow& \sum_{n=1}^N\alpha_nt_n &=& 0 \\
\frac{\partial{L}}{\partial{\xi}_n} = 0 &\Rightarrow& \alpha_n &=& C-\mu_n
\end{array}
\end{equation}$$

同理软间隔的对偶形式为

$$\begin{equation}
\begin{array}{rcl}
\hat{L}(\mathbf{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbf{x}_n, \mathbf{x}_m) \\
0 \le &\alpha_n& \le C \\
\sum_{n=1}^N\alpha_nt_n &=& 0
\end{array}
\end{equation}$$

我们发现软间隔与硬间隔的对偶形式是一致的，但是约束条件则由稍微不同。

## SMO算法

### 两个拉格朗日乘子的优化/求解

[SMO](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)算法是SVM中常用的一种求解算法，这个算法每次修改一对 $\alpha_i,\alpha_j$ 对，直到所有点都符合KKT条件，是对 [Osuna](http://webmail.svms.org/training/OsFG97.pdf)算法的一个改进。(之所以每次都只优化两个系数，是因为为了保持约束条件)。

首先 $\sum_{n=1}^N\alpha_nt_n = 0$ ,假设我们优化 $\alpha_1, \alpha_2$ ，那么 $t_1\alpha_1+t_2\alpha_2 = -\sum_{n=3}^N\alpha_nt_n=k$ ，由 $t_1、t_2$ 的取值不同性，可分为一下两种情况。（下面的k看做是常数）

![Lagrange multipliers](https://github.com/Lehyu/lehyu.cn/blob/master/image/PRML/svm/lm.png?raw=true)

考虑 $\alpha_2$ 的上下界 $[L,H]$

（1）当 $t_1 \ne t_2$ 时，$L=\max(0, \alpha_2-\alpha_1), H=min(C, C+\alpha_2-\alpha_1)$。 其中 L 对应上图左图中下方的两个红点， H对应上方的两个红点。

(2) 同理 $t_1 = t_2$ 时，$L=\max(0, \alpha_2+\alpha_1-C), H=min(C, \alpha_2+\alpha_1)$

对于 $\alpha_2$ 有

$$\begin{equation}
\begin{array}{rcl}
\alpha_2^{\text{new,unc}} &=& \alpha_2^{\text{old}} +\frac{t_2(E_1-E_2)}{\eta} \\
\eta &=& K_{11} + K_{12} - 2K_{12} = \|\phi(\mathbf{x}_1)-\phi(\mathbf{x}_2)\|^2
\end{array}
\end{equation}$$

证明：

$$\begin{equation}
\begin{array}{rcl}
y(\mathbf{x}) &=& \sum_{n=1}^N\alpha_nt_nK(\mathbf{x}, \mathbf{x}_n)+b \\
v_i &=& \sum_{n=3}^N\alpha_nt_nK(\mathbf{x}_i, \mathbf{x}_n) = y(\mathbf{x}_i) - \sum_{n=1}^2\alpha_nt_nK(\mathbf{x}_i, \mathbf{x}_n),i=1,2 \\
\text{目标函数： } \hat{L}(\mathbf{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbf{x}_n, \mathbf{x}_m) \\
\text{那么对于}\alpha_1,\alpha_2\text{: }\hat{L}(\alpha_1, \alpha_2) &=& \alpha_1+\alpha_2+\sum_{n=3}^N\alpha_n - \frac{1}{2}\left\{K_{11}\alpha_1^2+K_{22}\alpha_2^2+2t_1t_2\alpha_1\alpha_2+2t_1v_1\alpha_1+2t_2v_2\alpha_2+\sum_{n=3}^N\sum_{m=3}^N\alpha_n\alpha_mt_nt_mK_{nm}) \right \}\\
&=& \alpha_1+\alpha_2-(\frac{1}{2}K_{11}\alpha_1^2+(\frac{1}{2}K_{22}\alpha_2^2+t_1t_2\alpha_1\alpha_2+t_1v_1\alpha_1+t_2v_2\alpha_2)+\text{constant} \\
\sum_{n=1}^Nt_n\alpha_n = 0 &\text{and} & t_n^2 = 1 \\
\hat{L}(\alpha_2) &=& -(\frac{1}{2}K_{11}(k-\alpha_2t_2)^2 -(\frac{1}{2}K_{22}\alpha_2-t_2K_{12}(k-\alpha_2t_2)\alpha_2 \\
&+& (k-\alpha_2t_2)t_1 + \alpha_2 - v_1(k-\alpha_2t_2) - t_2v_2\alpha_2 +\text{constant} \\
\frac{\partial{\hat{L}}}{\partial\alpha_2} &=& -K_{11}\alpha_2 - K_{22}\alpha_2 + 2K_{12}\alpha_2 + K_{11}kt_2-K_{12}kt_2 - t_1t_2+1+v_1t_2-v_2t_2 \\
(K_{11} +K_{22}-2K_{12})\alpha_2 &=& t_2(t_2-t_1+kK_{11}-kK_{12}+v_1-v_2) \\
\alpha_2^{\text{new,unc}} &=& \alpha_2^{\text{old}} +\frac{t_2(E_1-E_2)}{\eta}
\end{array}
\end{equation}$$

由于 $\alpha$ 有上下界,因此

$$\begin{equation}
\begin{array}{rcl}
\alpha_2^{\text{new,clipped}} &=& \begin{cases}
H, & \alpha_2^{\text{new,unc}} \ge H \cr
\alpha_2^{\text{new,unc}}, & L \lt \alpha_2^{\text{new,unc}} \lt H \cr
L, & \alpha_2^{\text{new,unc}} \le L
\end{cases}
\end{array}
\end{equation}$$

为了保持约束条件,因此 $\alpha_1$ 的更新跨度与 $\alpha_2$ 一致, 方向相反.

$$\begin{equation}
\begin{array}{rcl}
\alpha_1^{\text{new}} &=& \alpha_1 + s(\alpha_2 - \alpha_2^{\text{new,clipped}})
\end{array}
\end{equation}$$

### 如何选择拉格朗日乘子对

我们希望最终所有拉格朗日乘子都不违反 KKT 条件, 因此我们希望每次选择乘子对时, 必须有一个乘子违反了 KKT 条件.

```
KKT: 1) a = 0     and t*y >= 1
     2) 0 < a < C and t*y = 1
     3) a = C     and t*y <= 1
find alphas[ix] which violate KKT: r2 = t*y-1, r2 < 0 ,r2 = 0 or r2 > 0
     1) r2 < 0 if a < C, violate
     2) r2 = 0 , none
     3) r2 > 0 if a > 0, violate
```

首先找到违反KKT条件的乘子, 然后我们希望下一个乘子能使优化的步长最大. 由 $\alpha_2$ 的更新公式我们知道 $|E_1-E_2|$ 最大时,即我们所选取的第二个乘子.

## SVM实现

有兴趣的可以参考我的[github](https://github.com/Lehyu/pyml/blob/master/models/svm/SVC.py)上实现的代码

## 参考
[1] [Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)

[2] PRML 第七章

[3] 统计学习方法-李航 第七章
