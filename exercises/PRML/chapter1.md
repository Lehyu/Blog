1.1

$$\begin{equation}
\begin{array}{rcl}
E(\mathbb{w}) &=& \frac{1}{2}\sum_{n=1}^N\{y(x_n, \mathbb{w}) - t_n\}^2 \\
y(x_n, \mathbb{w}) &=& \sum_{j=0}^M{w_jx_n^j} \\
\frac{dE}{dy} &=& \sum_{n=1}^N\{y(x_n, \mathbb{w})-t_n\} \\
\forall i,\frac{dE}{dw_i} &=& \sum_{n=1}^N\{y(x_n, \mathbb{w})-t_n\}\times x_n^i
\end{array}
\end{equation}$$

令 $\forall i,\frac{dE}{dw_i}=0$，得到

$$\begin{equation}
\begin{array}{rcl}
\sum_{n=1}^N\{\sum_{j=0}^M{w_jx_n^j}-t_n\}\times x_n^i=0
\end{array}
\end{equation}$$

将上式展开，得到 $\sum_{n=1}^N\sum_{j=0}^M{w_jx_n^{i+j}} =\sum_{n=1}^Nx_n^it_n$，即 $\sum_{j=0}^Mw_j(\sum_{n=1}^N{x_n^{i+j}}) =\sum_{n=1}^Nx_n^it_n$

1.2

1.2与1.1的解法是类似的，最后得到的结果是

$$\sum_{j=0}^Mw_j(\sum_{n=1}^N{x_n^{i+j}})+\lambda w_i =\sum_{n=1}^Nx_n^it_n$$

引入单位矩阵即可简化成

$$\sum_{j=0}^M(A_{ij}+\lambda I_{ij})w_j+ =\sum_{n=1}^Nx_n^it_n$$

1.3

考察的是全概率公式：$P(X)=\sum_YP(Y)P(X\vert Y)$，和贝叶斯公式：$P(Y\vert X)=\frac{P(X\vert Y)P(Y)}{P(X)}$

$$\begin{equation}
\begin{array}{rcl}
p(B=r) &=& 0.2, &p(B=b) &=& 0.2, &p(B=g) &=& 0.6 \\
p(F=a\vert B=r) &=& 0.3,&p(F=o\vert B=r) &=& 0.4, &p(F=l\vert B=r) &=& 0.3 \\
p(F=a\vert B=b) &=& 0.5, &p(F=o\vert B=b) &=& 0.5, &p(F=l\vert B=b) &=& 0 \\
p(F=a\vert B=g) &=& 0.3, &p(F=o\vert B=g) &=& 0.3, &p(F=l\vert B=g) &=& 0.4 \\
\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
p(F=a) &=& \sum_{B_i}{p(F=a,B=B_i)}=\sum_{B_i}{p(F=a\vert B=B_i)p(B=B_i)}=0.34 \\
p(B=g\vert F=o) &=&\frac{p(F=o\vert B=g)p(B=g)}{P(F=o)}=0.5
\end{array}
\end{equation}$$


1.4 考察连续概率分布的转换、最大密度。

在连续概率分布中，最大密度的点具有导数为0的性质。如果对于一个非概率分布的函数，$f(x),x=g(y)$，那么 $\tilde{f}(y)=f(g(y))$

$$\tilde{f'}(y)=f'(g(y))g'(y)=0$$

假如 $f(x)$ 具有在 $\hat{x}$ 最大值并且 $\hat{x}$ 不是边界点，那么 $f'(\hat{x})=0$ ，那么可以知道 $\hat{x}=g(\hat{y})$

但是在概率分布函数中，$p(y)=p_x(g(y))\vert g'(y)\vert $，令 $sg(y)=\vert g'(y)\vert ,s\in\{-1,+1\}$

那么 $p'(y)=sp_x'(g(y))(g'(y))^2+sp_x(g(y))g''(y)$

假设 $p_x(x)$ 在 $\hat{x}$ 具有最大密度，即 $p_x(\hat{x})=0$， 且 $\hat{x}=g(\hat{y})$，此时 $p'(\hat{y})=sp_x'(g(\hat{y}))(g'(\hat{y}))^2+sp_x(g(\hat{y}))g''(\hat{y})=sp_x(g(\hat{y}))g''(\hat{y})$

当 $x=g(y)$ 是非线性变换时，$g''(y)$ 不恒为0，故最大密度的变换与二阶导数相关。

当 $x=g(y)$ 是线性变换时，$g''(y)=0$，故最大密度的变换可由该线性变换得出。


1.5 考察的是期望与方差的基本概念与期望的一些基本性质: $E[C]=C,E[Cx]=CE[x]$


$$\begin{equation}
\begin{array}{rcl}
var[f] & = & E[\{f(x)-E[f(x)]\}^2]\\
 & = & E[f(x)^2-2f(x)E[f(x)+E[f(x)]^2]]\\
 & = & E[f(x)^2]-2E[f(x)]^2+E[f(x)]^2\\
 & = & E[f(x)^2]-E[f(x)]^2
\end{array}
\end{equation}$$

1.6 考察协方差与独立的概念

covariance:  $cov[x,y]=E_{x,y}[\{x-E[x]\}\{y-E[y]\}]=E_{x,y}[xy]-E[x]E[y]$
independent: $p(x,y)=p(x)p(y)$

$$E[xy]=\sum_i\sum_jx_iy_jp(x_i,y_j)=\sum_i\sum_jx_iy_jp(x_i)p(y_j)=\sum_jy_jp(y_j)\sum_ix_ip(x_i)=E[x]E[y]$$

1.7 考察指数积分以及正态分布的形式：$N(x\vert \mu,\sigma^2)=\frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$


$$\int{N(x\vert \mu,\sigma^2)}dx=\int{\frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})}dx=\frac{I}{\sqrt{2\pi \sigma^2}}=1$$

1.8 考察积分

积分区间关于原点对称的积分函数积分为0。

$$
\begin{equation}
\begin{array}{rcl}
E(x) & = & \int{x\frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})}dx\\
 & = & \int{(y+\mu)\frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{y^2}{2\sigma^2})}dy=\mu \\
 E[x^2] &=& \int{x^2N(x\vert \mu,\sigma^2)}dx\\
 &\overset{y=x-\mu}{=}& \int{(y^2+2y\mu+\mu^2)\frac{1}{\sqrt{2\pi \sigma^2}} \exp(-\frac{y^2}{2\sigma^2})}dy \\
 &=& \int{(y^2+\mu^2)\frac{1}{\sqrt{2\pi \sigma^2}} \exp(-\frac{y^2}{2\sigma^2})}dy \\
 &=& \sigma^2+\mu^2
\end{array}
\end{equation}$$

[应用分部积分：$\int{u}dv=uv-\int{v}du$ ]
故 $var[x]=E[x^2]-E[x]^2=\sigma^2$

1.9 这道题的思想与1.4有点类似，都是求mode，即求偏导为0的点。

考察向量与矩阵的求导

对于单变量: $\frac{dN}{dx}=-N(x\vert \mu,\sigma^2)\frac{x-\mu}{\sigma^2}=0$，解得 $\hat{x}=\mu$

对于多变量：$\frac{d\boldsymbol{N}}{d\boldsymbol{\mathbb{x}}}=-\boldsymbol{N}\boldsymbol\Sigma^{-1}(\boldsymbol{\mathbb{x}}-\boldsymbol{\mathbb{\mu}})=0$，解得 $\boldsymbol{\mathbb{\hat{x}}}=\boldsymbol{\mathbb{\mu}}$

1.10 考察概率独立、期望、方差的概念

$$\begin{equation}
\begin{array}{rcl}
E[x+z]&=&\int\int{(x+z)p(x,z)}dxdz\\
&=&\int\int{(x+z)p(x)p(z)}dxdz\\
&=&\int{xp(x)}dx+\int{zp(z)}dz\\
&=&E[x]+E[z] \\
E[(x+z)^2]&=&\int\int{(x+z)^2p(x,z)}dxdz\\
&=&\int\int{(x+z)^2p(x)p(z)}dxdz\\
&=&\int{x^2p(x)}dx+\int\int{2xzp(x)p(z)}dxdz+\int{z^2p(z)}dz\\
&=&E[x^2]+2E[x]E[z]+E[z^2] \\
var[x+z]&=&E[(x+z)^2]-E[x+z]^2=var[x]+var[z]
\end{array}
\end{equation}$$

1.11 简单的求导

1.12 考察样本抽取的性质(独立同分布)

当 $n=m$ 时，$E[x_nx_m]=E[x_n^2]=\mu^2+\sigma^2$

当 $n\neq m$时， $x_n,x_m$ 独立同分布，故 $E[x_nx_m]=\int\int{x_nx_mN^2}dx_ndx_m=\int{x_mN}dx_m\int{x_nN}dx_n=\mu^2$

$$E[\mu_{ML}]=E[\frac{1}{N}\sum_{n=1}^N{x_n}]=\frac{1}{N}\sum_{n=1}^2E[x_n]=\mu$$

$$\begin{equation}
\begin{array}{rcl}
E[\sigma_{ML}^2]&=&E[\frac{1}{N}\sum_{n=1}^N(x_n-\mu_{ML})^2] \\
&=&\frac{1}{N}\sum_{n=1}^NE[(x_n-\mu_{ML})^2] \\
&=&\frac{1}{N}\sum_{n=1}^NE[(x_n^2-\frac{2}{N}x_n\sum_{m=1}^Nx_m+\frac{1}{N^2}\sum_{i=1}^N\sum_{j=1}^Nx_ix_j)] \\
&=& \mu^2+\sigma^2-\frac{2}{N^2}(N^2\mu^2+N\sigma^2)+(\mu^2+\frac{1}{N}\sigma^2) \\
&=&\frac{N-1}{N}\sigma^2
\end{array}
\end{equation}$$

1.13 与上题类似

$$\begin{equation}
\begin{array}{rcl}
E[\sigma_{ML}^2]&=&E[\frac{1}{N}\sum_{n=1}^N(x_n-\mu)^2] \\
&=&\frac{1}{N}\sum_{n=1}^NE[(x_n-\mu)^2] \\
&=&\frac{1}{N}\sum_{n=1}^NE[(x_n^2-2x_n\mu+\mu^2)] \\
&=& \mu^2+\sigma^2-2\mu^2+\mu^2\\
&=&\sigma^2
\end{array}
\end{equation}$$

1.14

对于任意矩阵，$w_{ij}^S=(w_{ij}+w_{ji})/2,w_{ij}^A=(w_{ij}-w_{ji})/2$

$$\sum_{i=1}^D\sum_{j=1}^Dw_{ij}x_ix_j=\sum_{i=1}^D\sum_{j=1}^Dw_{ij}^Sx_ix_j+\sum_{i=1}^D\sum_{j=1}^Dw_{ij}^Ax_ix_j$$

如果 $w_{ij}$ 对称，那么 $w_{ij}^A = 0$

明显只有 $D(D+1)/2$ 个独立的参数

1.15 考察推理能力

由1.14可以知道权值在M维空间上是对称的，那么由式(1.134) $M^{th}$ 项的独立系数可以得到

$$\begin{equation}
\begin{array}{rcl}
n(D,M)&=&\sum_{i_1=1}^D\sum_{i_2=1}^{i_1}\dots\sum_{i_M=1}^{i_{M-1}}1 \\
&=&\sum_{i_1=1}^D\{\sum_{i_2=1}^{i_1}\dots\sum_{i_M=1}^{i_{M-1}}1\}\\
&=&\sum_{i_1}^Dn(i,M-1)
\end{array}
\end{equation} \tag{1.15-1}$$

[数学归纳法](http://baike.baidu.com/link?url=oqT1D9lYc1us_NkdhFgTrWahP9s9wKmR4Pmz_JQgd2XwoAFMHXub8S2Bh-Luk6D-F69kFfzhRgdm8oJYwqpSq_)

(1)证： $\sum_{i=1}^D\frac{(i+M-2)!}{(i-1)!(M-1)!}=\frac{(D+M-1)!}{(D-1)!M!}$
当 $D=1$ 时，原式成立；假设 $D=n$ 时，原式成立，即

$$\sum_{i=1}^n\frac{(i+M-2)!}{(i-1)!(M-1)!}=\frac{(n+M-1)!}{(n-1)!M!} \tag{1.15-2}$$

当 $D=n+1$ 时，

$$\begin{equation}
\begin{array}{rcl}
\sum_{i=1}^{n+1}\frac{(i+M-2)!}{(i-1)!(M-1)!}&=&\sum_{i=1}^n\frac{(i+M-2)!}{(i-1)!(M-1)!}+\frac{(n+1+M-2)!}{(n+1-1)!(M-1)!} \\
&=& \frac{(n+M-1)!}{(n-1)!M!}+\frac{(n+1+M-2)!}{(n+1-1)!(M-1)!}\\
&=&\frac{(n+M)!}{n!M!} \\
&=&\frac{(n+1+M-1)!}{(n+1-1)!M!}
\end{array}
\end{equation} \tag{1.15-1}$$

成立，故  $\sum_{i=1}^D\frac{(i+M-2)!}{(i-1)!(M-1)!}=\frac{(D+M-1)!}{(D-1)!M!}\tag{1.15-3}$

(2)证：$n(D,M)=\frac{(D+M-1)!}{(D-1)!M!}$

当 $M=2$ 时， $n(D,2)=D(D+1)/2$，显然成立；假设 $M=m$ 时，成立，即有
$$n(D,M-1)=\frac{(D+m-1)!}{(D-1)!m!}$$
当 $M=m+1$ 时，结合式(1.15-1)和(1.15-3)，即可证


1.16

(1)归纳：$N(D,M)=\sum_{m=0}^Mn(D,m)$

$$\begin{equation}
\begin{array}{rcl}
N(D,M) &=& \sum_{m=0}^M\sum_{i_1=1}^D\sum_{i_2=1}^{i_1}\dots\sum_{i_m=1}^{i_{m-1}}1 \\
&=&\sum_{m_0}^Mn(D,m)
\end{array}
\end{equation} \tag{1.16-1}$$

(2)证： $N(D,M)=\frac{(D+M)!}{D!M!}$

当 $M=0$ 时，$N(D,0)=n(D,0)=\frac{(D-1)!}{(D-1)!}=\frac{(D+0)!}{D!0!}$ 成立；假设 $M=m$ 时成立；当 $M=m+1$ 时，

$$\begin{equation}
\begin{array}{rcl}
N(D,m+1)&=&\sum_{i=0}^{m+1}n(D,i) \\
&=&\sum_{i=0}^mn(D,i)+n(D,m+1) \\
&=& \frac{(D+m)!}{D!m!}+\frac{(D+m+1-1)!}{(D-1)!(m+1)!} \\
&=& \frac{(D+m+1)!}{D!(m+1)!}
\end{array}
\end{equation}$$

成立，故  $N(D,M)=\frac{(D+M)!}{D!M!}$

(3)由Stirling's approximation: $n!\simeq n^ne^{-n}$，那么
$$N(D,M)=\frac{(D+M)!}{D!M!}\simeq\frac{(D+M)^-{(D+M)}e^{-(D+M)}}{D^De^{-D}M^Me^{-M}}\simeq D^M$$

1.17 考察[分部积分](http://baike.baidu.com/link?url=SgMfvZ69QHuAGnqIjXyanonBdZL6DV2EIfVFp0wY024Hby80pPq4U6q3cnDI_-9Q3RpCwjf5mt-iX1hcoEYRF9WIG5BmtVCyq-DyXzzoImVTKFCJRoj64aXImJsRM881sOIVhR2GUDNN6IUegYcg9q)： $\int udv=uv-\int vdu$

$$\begin{equation}
\begin{array}{rcl}
\Gamma(x+1)&=&\int_0^{\infty}u^xe^{-u}du \\
&=&-\int_0^{\infty}u^xde^{-u} \\
&=&-u^xe^{-u}+\int_0^{\infty}e^{-u}du^x \\
&=&x\int_0^{\infty}u^{ x-1}e^{-u}du \\
&=& x\Gamma(x)
\end{array}
\end{equation}$$

$\Gamma(1)=\int_0^{\infty}e^{-u}du=1$，故当x是整数时，有 $\Gamma(x+1)=x!$


1.18

由于 $\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}e^{x^2}e^{y^2}dxdy=\pi$，所以 $\prod_{i=1}^D\int_{-\infty}^{\infty}e^{x_i^2}dx_i = \pi^{(D/2)}$

$$\begin{equation}
\begin{array}{rcl}
S_D\int_0^{\infty}e^{-r^2}r^{D-1}dr &\overset{u=r^2}=& S_D\int_0^{\infty}{e^{-u} u^{\frac{D-1}{2} } u^{-\frac{1}{2} }/2 }du \\
&=& \frac{1}{2}S_D\int_0^{\infty}{e^{-u} u^{\frac{D}{2} -1 }  }du \\
&=&\frac{1}{2}S_D\Gamma(D/2)
\end{array}
\end{equation}$$

$\prod_{i=1}^D\int_{-\infty}^{\infty}e^{x_i^2}dx_i=S_D\int_0^{\infty}e^{-r^2}r^{D-1}dr \Rightarrow S_D=\frac{2\pi^{D/2}}{\Gamma(D/2)}$

$$V_D=\int_0^aS_Dr^{D-1}dr=\frac{S_Da^D}{D}$$

1.19

$\frac{V_S}{V_D}=\frac{\frac{S_D}{D} }{(2a)^D}=\frac{2\pi^{D/2}a^D}{D（2a)^D\Gamma(D/2)}=\frac{\pi^{D/2}}{D2^{D-1}\Gamma(D/2)}$

由Stirling's formula，得

$$\begin{equation}
\begin{array}{rcl}
\lim_{D\to\infty}\frac{V_S}{V_D}&=&\lim_{D\to\infty}\frac{\pi^{D/2}}{D2^{D-1}\Gamma(D/2)} \\
&\simeq&\lim_{D\to\infty}\frac{\pi^{D/2}}{D2^{D-1}(2\pi)^{1/2}e^{D/2}(D/2)^{D/2+1/2}} \\
&=& 0
\end{array}
\end{equation}$$

中心点到任意一个顶点的距离为： $d=\sqrt{\sum_i^D(2a)^2}/2=a\sqrt{D},ratio=\frac{d}{a}=\sqrt{D}$

1.20 空间想象力~~

$$V_s=\int_{shell}{p(x)}dx\simeq \frac{1}{(2\pi\sigma^2)^{D/2}}\exp(-\frac{r^2}{2\sigma^2})S_Dr^{D-1}\epsilon$$

$$\frac{dV_s}{dr}\propto \{(D-1)r^{D-2}-\frac{r^D}{\sigma^2}\}S_D\exp(-\frac{r^2}{2\sigma^2})=0 \Rightarrow\hat{r}=\sigma\sqrt{D-1}\simeq\sigma\sqrt{D}$$

这道题的结论应该 **仔细** 看看，**必须** 看懂。

1.21 考察决策论里错误率的概念: $p(mistake)=p(x\in \boldsymbol{R}_1,\boldsymbol{C}_2)+p(x\in \boldsymbol{R}_2,\boldsymbol{C}_1)=\int_{\boldsymbol{R}_1}{p(x,\boldsymbol{C}_2)}dx+\int_{\boldsymbol{R}_2}{p(x,\boldsymbol{C}_1)}dx$

在 $\boldsymbol{R}_1$ 空间 $p(x,\boldsymbol{C}_2)\leq p(x,\boldsymbol{C}_1)$；在 $\boldsymbol{R}_2$ 空间 $p(x,\boldsymbol{C}_1)\leq p(x,\boldsymbol{C}_2)$

那么

$$\begin{equation}
\begin{array}{rcl}
p(mistake)&\leq& \int_{\boldsymbol{R}_1}{(p(x,\boldsymbol{C}_1)p(x,\boldsymbol{C}_2))^{1/2}}dx+\int_{\boldsymbol{R}_2}{(p(x,\boldsymbol{C}_1)p(x,\boldsymbol{C}_2))^{1/2}}dx \\
&\leq& \int\{p(x,\boldsymbol{C}_1)p(x,\boldsymbol{C}_2)\}^{1/2}dx
\end{array}
\end{equation}$$

也就是说，误差率有个上界，只要我们减小这个上界，相应的误差也会减小

1.22 考察最大后验与期望风险

$L_{kj}$ 表示类k被赋予类j的损失值，那么

$$\begin{equation}
\begin{array}{rcl}
E[\boldsymbol{L}]&=&\sum_k\sum_j\int_{\boldsymbol{R}_j}L_{kj}p(\mathbb{x},\boldsymbol{C}_k)d\mathbb{x}\\
&=&\sum_j\int_{\boldsymbol{R}_j}\sum_kL_{kj}p(\mathbb{x},\boldsymbol{C}_k)d\mathbb{x}
\end{array}
\end{equation}$$

即对每个 $\mathbb{x}$ ，我们要最小化 $\sum_kL_{kj}p(\mathbb{x},\boldsymbol{C}_k)\propto \sum_kL_{kj}p(\boldsymbol{C}_k\vert \mathbb{x})$

当 $L_{kj}=1-I_{kj}$ 时， 样本分对 $L_{kj}=0$ ；样本分错 $L_{kj}=1$ ；也就是最小化误分率。此时相当于最小化 $1-p(\boldsymbol{C}_j\vert \mathbb{x})$ ，即最大化后验概率 $p(\boldsymbol{C}_j\vert \mathbb{x})$。

1.23 由最小化 $\sum_kL_{kj}p(\mathbb{x},\boldsymbol{C}_k)\propto \sum_kL_{kj}p(\boldsymbol{C}_k\vert \mathbb{x})$ 简单推导出 $\frac{1}{p(\mathbb{x})}\sum_kL_{kj}p(\mathbb{x}\vert \boldsymbol{C}_k)p(\boldsymbol{C}_k)$

1.24 考察rejected option： $p(C_k\vert x)<\theta, reject$

由题意知， $1-p(C_j\vert \mathbb{x})>\lambda,reject$ 即 $p(C_j\vert \mathbb{x})<1-\lambda,reject$ 。那么

由习题(1.22)可知，当 $L_{kj}=1-I_{kj}$ 时，最小化误分率相当于最大化后验概率。故 $\theta=1-\lambda$

1.25 考察一维到高维的推广

推导过程跟1.5.5小节是一致的。

$$\begin{equation}
\begin{array}{rcl}
\mathbb{E}[\boldsymbol{L}] &=& \int\int{\vert \vert \mathbb{y}(\mathbb{x})-\mathbb{t}\vert \vert ^2p(\mathbb{x},\mathbb{t})  }d\mathbb{x}d\mathbb{t} \\
\frac{\mathbb{E}[\boldsymbol{L}]}{d\mathbb{y}} &=& \int{2(\mathbb{y}(\mathbb{x})-\mathbb{t})p(\mathbb{x},\mathbb{t})  }d\mathbb{t} =0 \\
\mathbb{y}(\mathbb{x}) &=& \frac{\int{\mathbb{t}p(\mathbb{x},\mathbb{t})  }d\mathbb{t}}{p(\mathbb{x})}=\int{\mathbb{t}p(\mathbb{t}\vert \mathbb{x}) }d\mathbb{t}=\mathbb{E}_{\mathbb{t}}[\mathbb{t}\vert \mathbb{x}]
\end{array}
\end{equation}$$


1.26 主要考察的还是向量/矩阵的运算

推导过程可以参照1.5.5小节(1.90)的推导。

1.27 考察积分求导，与 $L_q$ 在 $q$ 取不同的值得时候的含义

$$E[L_q]=\int\int{\vert y(\mathbb{x})-t\vert ^qp(t,\mathbb{x})}d\mathbb{x}dt$$

令 $\frac{dE}{dy} = 0$

$$\begin{equation}
\begin{array}{rcl}
\frac{dE}{dy}&=&\int{q\vert y(\mathbb{x})-t\vert ^{q-1}sign(y(x),t)p(t,\mathbb{x})}d\mathbb{x}dt=0 \\
&\Rightarrow&\int_{-\infty}^{y(\mathbb{x})}{q\vert y(\mathbb{x})-t\vert ^{q-1}p(t,\mathbb{x})}dt-\int_{y(x)}^{\infty}{q\vert y(\mathbb{x})-t\vert ^{q-1}p(t,\mathbb{x})}dt=0 \\
&\Rightarrow&\int_{-\infty}^{y(\mathbb{x})}{\vert y(\mathbb{x})-t\vert ^{q-1}p(t,\mathbb{x})}dt=\int_{y(x)}^{\infty}{\vert y(\mathbb{x})-t\vert ^{q-1}p(t,\mathbb{x})}dt \\
&\overset{/p(x)}\Rightarrow&\int_{-\infty}^{y(\mathbb{x})}{\vert y(\mathbb{x})-t\vert ^{q-1}p(t\vert \mathbb{x})}dt=\int_{y(x)}^{\infty}{\vert y(\mathbb{x})-t\vert ^{q-1}p(t\vert \mathbb{x})}dt
\end{array}
\end{equation}$$

当 $q=1$ 时，$\int_{-\infty}^{y(\mathbb{x})}{p(t\vert \mathbb{x})}dt=\int_{y(x)}^{\infty}{p(t\vert \mathbb{x})}dt$ ，故 $y(\mathbb{x})$ 是条件中值。

当 $q\to0$ 时，$\vert y(\mathbb{x})-t\vert ^q\to1, \forall \vert y(x)-t\vert  \overset{not}\to 0$。故 $y(\mathbb{x})$ 是条件模

1.28 考察信息论里的基础概念: $h(x)=-\log_2{p(x)}$

(这里的 $p$ 应该是概率密度函数吧?如果理解没错的话，推导如下)

$$h(p^2)=-\log_2{p(x)^2}=-2\log_2{p(x)}=2h(p)$$

同理， $h(p^n)=nh(p),h(p^{n/m})=(n/m)h(p)$
那么 $h(p)\propto\ln{p}$

1.29 考察凸函数的性质

Jensen's inequality: $f(\sum_{i=1}^M\lambda_ix_i)\leq\sum_{i=1}^M\lambda_if(x_i),\sum_i\lambda_i=1$

entropy: $H[x]=-\sum_{x}p(x)\log_2{p(x)}$

令 $f(x)=\log_2x$，$f(x)$ 为凹函数，那么由Jensen不等式得到
$$\log_2(\sum_{i=1}^M\lambda_i\frac{1}{x_i})\geq\sum_{i=1}^M\lambda_i\log_2\frac{1}{x_i}$$

由于 $\sum_xp(x)=1$， 故 $\sum_{i=1}^Mp(x)\log_2\frac{1}{x_i}\leq\log_2(\sum_{x}p(x)\frac{1}{p(x)})$，即 $H[x]\leq\log_2M$

(对数的底可以任意取，用的较多的是 $\log_2,\ln$)

1.30 考察KL离散： $KL(p\vert \vert q)=-\int{p(\mathbb{x})\ln{\frac{q(\mathbb{x})}{p(\mathbb{x})}} }d\mathbb{x}$

当 $p(x)=\mathcal{N}(x\vert \mu,\sigma^2),q(x)=\mathcal{N}(x\vert m,s^2)$

$$\begin{equation}
\begin{array}{rcl}
KL(p\vert \vert q) &=& -\int{\mathcal{N}(x\vert \mu,\sigma^2)\ln\{ \frac{\frac{1}{\sqrt{2\pi s^2} }\exp(-\frac{(\mathbb{x}-m)^2}{2s^2}}{\frac{1}{\sqrt{2\pi\sigma^2} }\exp(-\frac{(\mathbb{x}-\mu)^2}{2\sigma^2}} \}}d\mathbb{x} \\
&=& -\int{\mathcal{N}(x\vert \mu,\sigma^2)\{\ln\sigma-\ln{s} -\frac{(\mathbb{x}-m)^2}{2s^2}+\frac{(\mathbb{x}-\mu)^2}{2\sigma^2}\}}d\mathbb{x} \\
&=& -\int{\mathcal{N}(x\vert \mu,\sigma^2)\{\ln\sigma-\ln{s} -\frac{(\mathbb{x}-m)^2}{2s^2}+\frac{(\mathbb{x}-\mu)^2}{2\sigma^2}\}}d\mathbb{x} \\
&=& -\int{\mathcal{N}(x\vert \mu,\sigma^2)\{\ln\sigma-\ln{s}\}}d\mathbb{x} + \int{\mathcal{N}(x\vert \mu,\sigma^2)\{\frac{(\mathbb{x}-m)^2}{2s^2}\}}d\mathbb{x}-
\int{\mathcal{N}(x\vert \mu,\sigma^2)\{\frac{(\mathbb{x}-\mu)^2}{2\sigma^2}\}}d\mathbb{x} \\
&=& \ln{s}-\ln{\sigma}+\frac{\sigma^2-s^2+(m-\mu)^2}{2s^2}
\end{array}
\end{equation}$$

1.31 考察信息论

$$\begin{equation}
\begin{array}{rcl}
H[\mathbb{x},\mathbb{y}] &=& -\int\int{p(\mathbb{x},\mathbb{y})\ln{p(\mathbb{x},\mathbb{y})} }d\mathbb{x}d\mathbb{y}  \\
&=& -\int\int{p(\mathbb{x}\vert \mathbb{y})p(\mathbb{y})\ln\{p(\mathbb{x}\vert \mathbb{y})p(\mathbb{y})\} }d\mathbb{x}d\mathbb{y}  \\
&=& -\int\int{p(\mathbb{x}\vert \mathbb{y})p(\mathbb{y})\ln\{p(\mathbb{x}\vert \mathbb{y})\} }d\mathbb{x}d\mathbb{y}-\int\int{p(\mathbb{x}\vert \mathbb{y})p(\mathbb{y})\ln\{p(\mathbb{y})\} }d\mathbb{x}d\mathbb{y}  \\
&=& H[\mathbb{x}\vert \mathbb{y}]+H[\mathbb{y}]  \\
d &=& -\int\int{ p(\mathbb{x},\mathbb{y})\ln{p(\mathbb{x},\mathbb{y})}  } d\mathbb{x}d\mathbb{y} + \int\int{ p(\mathbb{x},\mathbb{y})\ln\{p(\mathbb{x})p(\mathbb{y})\}  } d\mathbb{x}d\mathbb{y} \\
&=& -\int\int{ p(\mathbb{x},\mathbb{y})\ln{\frac{p(\mathbb{x})p(\mathbb{y})}{p(\mathbb{x},\mathbb{y})} }  } d\mathbb{x}d\mathbb{y}  \\
&=& KL(p(\mathbb{x},\mathbb{y})\vert \vert p(\mathbb{x})p(\mathbb{y})) \geq 0 \\
&=& H[\mathbb{x}]+H[\mathbb{y}]-H[\mathbb{x},\mathbb{y}] \geq 0
\end{array}
\end{equation}$$

故 $H[\mathbb{x},\mathbb{y}] \leq H[\mathbb{x}]+H[\mathbb{y}],H[\mathbb{x},\mathbb{y}] = H[\mathbb{x}]+H[\mathbb{y}]\iff p(\mathbb{x},\mathbb{y})=p(\mathbb{x})p(\mathbb{y})$


1.32 考察概率分布的转换: $p_y(y)=p_x(g(y))\vert g'(x)\vert ,x=g(y)$

$$\begin{equation}
\begin{array}{rcl}
\mathbb{y} &=& \mathbb{Ax} \\
p(\mathbb{x}) &=& p(\mathbb{y})\mathbb{\vert A\vert } \\
H[\mathbb{y}] &=& -\int{p(\mathbb{y})\ln{p(\mathbb{y})} }d\mathbb{y} \\
&=& -\int{p(\mathbb{x})\mathbb{\vert A\vert }^{-1}\ln\{p(\mathbb{x})\mathbb{\vert A\vert }^{-1} \} } \mathbb{\vert A\vert }d\mathbb{x} \\
&=& -\int{p(\mathbb{x})\ln\{p(\mathbb{x}) \} } d\mathbb{x}+\int{p(\mathbb{x})\ln\{\mathbb{\vert A\vert } \} } d\mathbb{x} \\
&=& H[\mathbb{x}]+\ln\mathbb{\vert A\vert }
\end{array}
\end{equation}$$

1.33

$$\begin{equation}
\begin{array}{rcl}
H[\mathbb{y}\vert \mathbb{x}] &=& 0 \\
 -\sum_{i}\sum_{j}{p(x_i,y_j)\ln\{p(y_j\vert x_i)\} } &=& 0 \\
\sum_{i}\sum_{j}{p(y_j\vert x_i)p(x_i)\ln\{p(y_j\vert x_i)\} } &=& 0 \\ \tag{1.33.1}
\end{array}
\end{equation}$$

并且 $p(x) > 0 \ \& \ p(y_j\vert x_i)\ln{p(y_j\vert x_i)} \leq 0 \\ \tag{1.33.2}$

$$\begin{equation}
\begin{array}{rcl}
&\Rightarrow& p(y_j\vert x_i)\ln{p(y_j\vert x_i)} = 0 \\
&\Rightarrow& p(y_j\vert x_i)=0 &or& p(y_j\vert x_i) = 1 \\
&and&\sum_jp(y_j\vert x_i)=1 \\
\end{array}
\end{equation}$$
存在唯一的一个值 $y_j$，使得 $p(y_j\vert x_i)=1,p(y_k\vert x_i)=0(k\neq j)$


1.34


1.35
$$\begin{equation}
\begin{array}{rcl}
\int_{-\infty}^{\infty}\mathcal{N}(x\vert \mu,\sigma^2)dx &=& 1 \\
\int_{-\infty}^{\infty}x\mathcal{N}(x\vert \mu,\sigma^2)dx &=& \mu \\
\int_{-\infty}^{\infty}(x-\mu)^2\mathcal{N}(x\vert \mu,\sigma^2)dx &=& \sigma^2 \\
H[x] &=& -\int{\mathcal{N}\ln\mathcal{N}}dx \\
&=& \int{\mathcal{N}\ln\sqrt{2\pi\sigma^2} }dx +\int{\mathcal{N}\frac{(x-\mu)^2}{2\sigma^2} }dx \\
&=&  \ln\sqrt{2\pi\sigma^2} + \frac{1}{2}\\
\end{array}
\end{equation}$$


1.36


1.37 参考1.31


1.38

当 M=1时成立，假设 $M=k$ 时成立，即 $f(\sum_{i=1}^k\lambda_ix_i) \leq \sum_{i=1}^k\lambda_if(x_i)$；当 $M=k+1$ 时
$$\begin{equation}
\begin{array}{rcl}
f(\lambda_{k+1}x_{k+1}+\sum_{i=1}^k\lambda_i'x_i) &\overset{\lambda_i'=\frac{\lambda_i}{1-\lambda_{k+1}} } \leq& \lambda_{k+1}f(x_{k+1})+(1-\lambda_{k+1})f(\sum_{i=1}^k\lambda_ix_i) \\
&\leq& \lambda_{k+1}f(x_{k+1})+\sum_{i=1}^k\lambda_i'f(x_i)
\end{array}
\end{equation}$$

1.39


1.40

1.41



























fdfd
