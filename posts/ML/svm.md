# Support Vector Machine

## SVM 最大间隔分类器

[线性分类器](chapter4_prml.md)中我们知道线性模型的一般形式为 $y(\mathbb{x})=\mathbb{w}^T\phi(\mathbb{x})+b$ ，对于二分类假设我们令 $t_n \in \{-1, 1\}$ ，那么对于 $t_n=1$ 的点有 $y(\mathbb{x}_n) > 0$，反之 $y(\mathbb{x}_n) < 0$ ， 所以我们有 $t_ny(\mathbb{x}) >0$ 。

那么任意一点 $\mathbb{x}$ 到分类平面的距离为 $\frac{t_ny(\mathbb{x}_n)}{\|\mathbb{w}\|}$ ， 我们知道在分类平面可能存在无数个，但是直观上看，当两个类别的点到分类平面的距离最大时，这个分类平面显然时最好的。

因此我们就希望优化一下目标函数

$$\begin{equation}
\begin{array}{rcl}
\arg\max_{\mathbb{w},b}\left\{ \frac{1}{\|\mathbb{w}\|}\min_n\left[ t_n(\mathbb{w}^T\phi(\mathbb{x}_n)+b )\right] \right\}
\end{array}
\end{equation}$$

对于上式，由于 $\mathbb{w}、b$ 可以缩放， 因此不失一般性的， 我们可以令 $t_n(\mathbb{w}^T\phi(\mathbb{x}_n)+b ) \ge 1$ ， 当 $t_n(\mathbb{w}^T\phi(\mathbb{x}_n)+b ) = 1$ 时， $(\mathbb{x}_n, t_n)$ 及为支持向量。所以优化目标函数可以转变为在约束条件 $t_n(\mathbb{w}^T\phi(\mathbb{x}_n)+b ) \ge 1$ 下，求

$$\begin{equation}
\begin{array}{rcl}
\arg\min_{\mathbb{w},b}\frac{1}{2}\|\mathbb{w}\|^2
\end{array}
\end{equation}$$

由拉格朗日乘子法得

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{w}, b, \mathbb{\alpha}) = \frac{1}{2}\|\mathbb{w}\|^2 - \sum_{n=1}^N\alpha_n\left\{ t_n(\mathbb{w}^T\phi(\mathbb{x}_n)+b )- 1 \right\}
\end{array}
\end{equation}$$

对 $\mathbb{w}, b$ 对导得

$$\begin{equation}
\begin{array}{rcl}
\mathbb{w} &=& \sum_{n=1}^N\alpha_nt_n\phi(\mathbb{x}_n) \\
0 &=& \sum_{n=1}^N\alpha_nt_n
\end{array}
\end{equation}$$

从而得到最大间隔问题的对偶形式

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0
\end{array}
\end{equation}$$

要是上述目标函数最优，必须满足 [KKT 条件](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)

$$\begin{equation}
\begin{array}{rcl}
\alpha_n &\ge& 0 \\
t_ny(\mathbb{x}_n)-1 &\ge& 0 \\
\alpha_n\{ t_ny(\mathbb{x}_n)-1  \} &=& 0
\end{array}
\end{equation}$$

## 软间隔



$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
L(\mathbb{\alpha}) &=& \sum_{n=1}^N\alpha_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\alpha_n\alpha_mt_nt_mk(\mathbb{x}_n, \mathbb{x}_m) \\
\alpha_n &\ge& 0 \\
\sum_{n=1}^N\alpha_nt_n &=& 0

\end{array}
\end{equation}$$
