## 2.1 考察熵与伯努利分布

熵： $H[x]=-\sum_{x} p(x)\ln p(x)$

伯努利分布：$\text{Bern}(x\vert \mu)=\mu^x(1-\mu)^{1-x}$

$$\begin{equation}
\begin{array}{rcl}
\sum_{x\in\{0,1\}} &=&\mu+1-\mu =1 \\
E[x] &=& \mu\\
E[x^2] &=& \mu \\
var[x] &=& E[x^2]-E[x]^2 = \mu(1-\mu)\\
H[x] &=& -\sum_{x=0}^1 \text{Bern}(x\vert \mu)\ln\text{Bern}(x\vert \mu) \\
&=& -\mu\ln\mu-(1-\mu)\ln(1-\mu)
\end{array}
\end{equation}$$

## 2.2 概率分布的基本定义

$$\begin{equation}
\begin{array}{rcl}
p(x=1\vert \mu)+p(x=-1\vert \mu) &=& \frac{1+\mu}{2}+\frac{1-\mu}{2} =1 \text{  所以标准化}\\
E[x] &=& p(x=1\vert \mu)-p(x=-1\vert \mu)=\mu \\
E[x^2] &=& p(x=1\vert \mu)+p(x=-1\vert \mu)=1 \\
var[x] &=& E[x^2] - E[x]^2 = 1-\mu^2 \\
H[x] &=& -p(x=1\vert \mu)\ln p(x=1\vert \mu) - p(x=-1\vert \mu)\ln p(x=-1\vert \mu) \\
&=& -\frac{1+\mu}{2}\ln\frac{1+\mu}{2}-\frac{1-\mu}{2}\ln\frac{1-\mu}{2}
\end{array}
\end{equation}$$

## 2.3 二项分布

$$\begin{equation}
\begin{array}{rcl}
\text{Bin}(x\vert N,\mu) &=& \binom{N}{m}\mu^m(1-\mu)^{N-m} \\
\binom{N}{m} &=& \frac{N!}{(N-m)!m!}
\end{array}
\end{equation}$$

### $\binom{N}{m} + \binom{N}{m-1}=\binom{N+1}{m}$

$$\begin{equation}
\begin{array}{rcl}
\binom{N}{m} + \binom{N}{m-1} &=& \frac{N!}{(N-m)!m!} + \frac{N!}{(N-m+1)!(m-1)!} \\
&\overset{\text{同分母}}=& \frac{N!(N-m+1)}{(N-m)!m!(N-m+1)} + \frac{N!m}{(N-m+1)!(m-1)!m} \\
&=& \frac{(N+1)!}{(N+1-m)!m!} \\
&=& \binom{N+1}{m}
\end{array}
\end{equation}$$

### $(1+x)^N=\sum_{m=0}^N\binom{N}{m}x^m$
当N=1的时候，成立；假设当N=k的时候成立，即 $(1+x)^k=\sum_{m=0}^k\binom{k}{m}x^m$ ，当 $N=k+1$ 时，有

$$\begin{equation}
\begin{array}{rcl}
(1+x)^{k+1} &=& (1+x)^k(1+x) \\
&=& \sum_{m=0}^k\binom{k}{m}x^m(1+x) \\
&=& \sum_{m=0}^k\binom{k}{m}x^m +\sum_{m=1}^k\binom{k}{m}x^m \\
&=& 1+\sum_{m=1}^{k-1}\left(\binom{k}{m}+\binom{k}{m-1}\right)x^m +x^{k+1}\\
&=& \sum_{m=0}^{k+1}\binom{k+1}{m}x^m \\
\Rightarrow (1+x)^N &=& \sum_{m=0}^N\binom{N}{m}x^m
\end{array}
\end{equation}$$

### $\text{Bin}(x\vert N,\mu)=\binom{N}{m}\mu^m(1-\mu)^{N-m}$ 标准化

$$\begin{equation}
\begin{array}{rcl}
\sum_{m=0}^N\text{Bin}(x\vert N,\mu) &=& \sum_{m=0}^N\binom{N}{m}\mu^m(1-\mu)^{N-m} \\
&=& (\mu+1-\mu)^N =1
\end{array}
\end{equation}$$

## 2.4

二项分布的期望为 $E[m]=\sum_{m=0}^Nm\text{Bin}(m\vert N,\mu)$，直接求比较难，所以我们先求 $\sum_{m=0}^N\text{Bin}(x\vert N,\mu) = \sum_{m=0}^N\binom{N}{m}\mu^m(1-\mu)^{N-m} =1$ 关于 $\mu$ 的导数，即

$$\begin{equation}
\begin{array}{rcl}
\sum_{m=0}^N\binom{N}{m}\mu^{m-1}(1-\mu)^{N-m}m &-&\sum_{m=0}^N\binom{N}{m}\mu^{m-1}(1-\mu)^{N-m-1}(N-m) &=& 0 \\
\sum_{m=0}^N\binom{N}{m}\mu^{m-1}(1-\mu)^{N-m}m &=& \sum_{m=0}^N\binom{N}{m}\mu^m(1-\mu)^{N-m-1}(N-m) \\
\overset{\text{同乘 }\mu(1-\mu)}\Rightarrow \\
\sum_{m=0}^N\binom{N}{m}\mu^{m}(1-\mu)^{N-m}m(1-\mu) &=& \sum_{m=0}^N\binom{N}{m}\mu^m(1-\mu)^{N-m}(N-m)\mu \\
\Rightarrow E[m] &=& N\mu
\end{array}
\end{equation}$$

求方差前可以先求 $E[m^2]$，明显这个和 $E[m]$ 的求法类似，二阶导

$$\begin{equation}
\begin{array}{rcl}
E[m^2] &=& N\mu(1-\mu)+N^2\mu^2 \\
\Rightarrow var[m] &=& N\mu(1-\mu)
\end{array}
\end{equation}$$

## 2.5 考察积分

[不太会]

## 2.6 基础定义

$$\begin{equation}
\begin{array}{rcl}
E[\mu] &=& \int_0^1\mu\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}d\mu \\
&=& \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\times\frac{\Gamma(a+1)\Gamma(b)}{\Gamma(a+1+b)} \\
\text{因为 }\Gamma(x+1) &=& x\Gamma(x)  \\
E[\mu] &=& \frac{a}{a+b} \\
E[\mu^2] &=& \int_0^1\mu^2\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}d\mu \\
&=& \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\times\frac{\Gamma(a+2)\Gamma(b)}{\Gamma(a+2+b)} \\
&=& \frac{a(a+1)}{(a+b)(a+b+1)} \\
\Rightarrow var[\mu] &=& \frac{ab}{(a+b)^2(a+b+1)} \\
\text{对贝塔分布求导，并令导数为0} \\
\text{mode}[\mu] &=&  \frac{a-1}{(a+b-2)}
\end{array}
\end{equation}$$

## 2.7 考察二项分布与beta分布

$$\begin{equation}
\begin{array}{rcl}
\text{Beta}(\mu\vert a,b) &=& \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1} \\
E[\mu] &=& \frac{a}{a+b}
\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
p(\mu\vert m,l) \propto \mu^m(1-\mu)^l\mu^{a-1}(1-\mu)^{b-1}
\end{array}
\end{equation}$$


所以后验仍然是贝塔分布，那么后验的期望是 $\frac{a+m}{a+b+m+l}$，而最大似然估计 $\mu_{ML}=\frac{m}{m+l}$ ，**要证 $\frac{a+m}{a+b+m+l} \in[\frac{a}{a+b}, \frac{m}{m+l}]$，即证 $\exists \lambda\in[0,1],\text{s.t. } \lambda\frac{a}{a+b}+(1-\lambda)\frac{m}{m+l}=\frac{a+m}{a+b+m+l}$**，计算得 $\lambda=\frac{1}{1+(m+l)/(a+b)}$

## 2.8 考察定义与基本计算

$$\begin{equation}
\begin{array}{rcl}
E[x] &=& \sum_{x}xp(x) \\
E_x[x\vert y] &=& \sum_{x}xp(x\vert y) \\
E_y[E_x[x\vert y]] &=& \sum_{y}p(y)\sum_{x}xp(x\vert y) \\
&=& \sum_{x}x\sum_{y}p(y)p(x\vert y) = E[x]\\
\\
E_y[var_x[x\vert y]]+var_y[E_x[x\vert y]] &=& E_y[E_x[x^2\vert y]-E[x\vert y]^2] \\
&& +E_y[E_x[x\vert y]^2]-E_y[E_x[x\vert y]]^2 \\
&=& E_x[x^2] - E_x[x]^2 \\
&=& var_x[x]
\end{array}
\end{equation}$$

## 2.9

$$\begin{equation}
\begin{array}{rcl}

\end{array}
\end{equation}$$

$$\begin{equation}
\begin{array}{rcl}
 \\
E[m] &=& N\mu \\
var[m] &=& N\mu(1-\mu)
\end{array}
\end{equation}$$



$$\begin{equation}
\begin{array}{rcl}
 \\
E[m] &=& N\mu \\
var[m] &=& N\mu(1-\mu)
\end{array}
\end{equation}$$



$$\begin{equation}
\begin{array}{rcl}
 \\
E[m] &=& N\mu \\
var[m] &=& N\mu(1-\mu)
\end{array}
\end{equation}$$


$$\begin{equation}
\begin{array}{rcl}
 \\
E[m] &=& N\mu \\
var[m] &=& N\mu(1-\mu)
\end{array}
\end{equation}$$


$$\begin{equation}
\begin{array}{rcl}
 \\
E[m] &=& N\mu \\
var[m] &=& N\mu(1-\mu)
\end{array}
\end{equation}$$
