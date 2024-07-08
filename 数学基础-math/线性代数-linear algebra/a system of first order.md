**the solution of constant coefficient linear equation are exponentials**

### example :

$$
\begin{align}
\frac{ d ^{} u_{1} }{ d  t }&=-u_{1}+2u_{2} \\
\frac{ d ^{} u_{2} }{ d t ^{} }  &= u_{1}-2u_{2}
\end{align}
$$
#### and matrix is :
$$
\begin{bmatrix}
-1 & 2 \\
1 & -2 \\
\end{bmatrix}
$$

#### is a singular matrix, so one of eigenvalues is 0, another is -3
$$
\lambda_{1}=0,\lambda_{2}=-3
$$
#### eigenvectors is :
$$
\begin{align}
\begin{bmatrix}
-1 -0 & 2 \\
1 & -2-0
\end{bmatrix}\vec{x}_{1} & =0 \\
\begin{bmatrix}
-1 & 2 \\
1 & -2
\end{bmatrix}\begin{bmatrix}
x_{1} \\
x_{2}
\end{bmatrix} &= 0 \\
\vec{x}_{1} &= \begin{bmatrix}
2 \\
1
\end{bmatrix} \\ 
\vec{x}_{2} &= \begin{bmatrix}
1 \\
-1
\end{bmatrix}
\end{align}
$$

#### the solution  is :

$$
\begin{align} \\
u(t) &= c_{1}e^{ \lambda_{1}t }\vec{x}_{1}+c_{2}e^{ \lambda_{2}t }\vec{x}_{2} \\ 
\end{align}
$$

##### check :
$$
\begin{align}
suppose:u(t)&=e^{ \lambda_{1}t }\vec{x}_{1}  \\
\frac{ d {} u }{ d t ^{} }  &= \mathbf{A}u \\
\lambda_{1}e^{ \lambda_{1}t }\vec{x}_{1} &= \mathbf{A} u \\
\lambda_{1}u &= \mathbf{A}u
\end{align}
$$
#### calculate c1,c2
$$
given:u_{0}=\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$
at $t=0$  
$$\begin{align} \\
c_{1}\vec{x}_{1}+c_{2}\vec{x}_{2} &= u_{0} \\

c_{1}\begin{bmatrix}
2 \\
1
\end{bmatrix}+c_{2}\begin{bmatrix}
1\\
-1
\end{bmatrix} &= \begin{bmatrix}
1 \\
0
\end{bmatrix} \\ \\
\begin{cases}
c_{1} &= \frac{1}{3} \\
c_{2} &= \frac{1}{3} \\
\end{cases} \\
u(t) &= \frac{1}{3}e^{ 0t }\vec{x}_{1}+\frac{1}{3e}e^{ -3t }\vec{x}_{2} \\
&= \frac{1}{3}e^{ 0t }\begin{bmatrix}
2 \\
1
\end{bmatrix}+\frac{1}{3e}e^{ -3t }\begin{bmatrix}
1 \\
-1
\end{bmatrix}
\end{align}$$


#### properties :
##### stability
- if all eigenvalues is negative, $u(t)\to_{0}$ , differential equation is stability 
- if there is complex, only the real part take effect 

prove :

$$\begin{align} \\
suppose:u(t) &= e^{ (-3+6i )t} \\
\lvert u(t) \rvert  &= \lvert ~~~ e^{ (-3+6i)t }  \rvert  \\
 &= \lvert e^{ -3t } \rvert \cdot \lvert e^{ 6it } \rvert  \\
 &= e^{ -3t }
\end{align}$$
##### steady state
- if there is a zero eigenvalue and other eigenvalue will disappear 

##### blow up
- if there are some plus eigenvalues , the equation is blow up
- the eigenvalues of $-\mathbf{A}$ is opposite to the eigenvalues of  $\mathbf{A}$ 

#### for $2\times 2$ 
$$\begin{align}
suppose:\mathbf{A} &= \begin{bmatrix}
a & b \\
c & d
\end{bmatrix} \\
\begin{cases}
a+d<0 \\
\det \mathbf{A}>0
\end{cases}
\end{align}$$
- A is convergent

#### summary 
for:

$$\begin{align}
 \frac{ d {} u }{ d t ^{} }  &= \mathbf{A}\vec{u} \\
\end{align}$$
matrix A couple u1 and u2, set
$$\begin{align}
\vec{u} &= \mathbf{S}\vec{v} \\
\mathbf{S}\frac{ d {} v }{ d t ^{} }  &= \mathbf{A}\mathbf{S}\vec{v} \\
\frac{ d {} v }{ d t ^{} }  &= \mathbf{S}^{-1}\mathbf{A}\mathbf{S}\vec{v} \\
 &= \Lambda \vec{v}
\end{align}$$
there is no couple

$$\begin{align}
\frac{ d {} v_{1} }{ d t ^{} }  &= \lambda_{1}\vec{v}_{1} \\
\dots
\end{align}$$
get v(t)
$$\begin{align}

\end{align}$$
$$\begin{align}
v(t) &= e^{ \Lambda t }v(0) \\
\end{align}$$
#### we want prove:
$$\begin{align}
u(t) &= \mathbf{S}e^{ \Lambda t }\mathbf{S}^{-1}u(0) = e^{ \mathbf{A}t }u(0) 
\end{align}$$

##### matrix exponential $e^{ \mathbf{A}t }$ 
$$\begin{align}
e^{ x } &= \sum_{i=0}^{N} \frac{x^{i}}{i!} \\
\frac{1}{1-x} &= \sum_{i=0}^{N} x^{i} \\
e^{ \mathbf{A}t } &= \mathbf{I}+\mathbf{A}t+\frac{(\mathbf{A}t)^{2}}{2}+\dots \\ \\
(\mathbf{I}-\mathbf{A}t)^{-1} &= \mathbf{I}+\mathbf{A}t +(\mathbf{A}t)^{2}+\dots
\end{align}$$
and $e^{ x }$ is better than $\frac{1}{1-x}$ ,because $e^{ x }$ is always convergent
go on to prove:
$$\begin{align}
e^{ \mathbf{A}t } &= \mathbf{I}+\mathbf{A}t+\frac{(\mathbf{A}t)^{2}}{2}+\dots \\ 
 &= \mathbf{S}\mathbf{S}^{-1} +\mathbf{S}\Lambda \mathbf{S}^{-1}t+\frac{\mathbf{S}\Lambda ^{2}\mathbf{S}}{2}t^{2}+\dots \\
 &= \mathbf{S}e^{ \Lambda t }\mathbf{S}^{-1}
\end{align}$$
##### what is $e^{ \Lambda t }$ 
$$\begin{align}
\Lambda &= \begin{bmatrix}
\lambda_{1} & 0 & \dots \\
\dots \\
0 & 0 & \lambda _{n}
\end{bmatrix} \\
e^{ \Lambda t } &= \begin{bmatrix}
e^{ \lambda_{1}t } & 0 & \dots \\
\dots \\
0 & 0 & e^{ \lambda _{n}t }
\end{bmatrix}
\end{align}$$
f th- if the real part of the eigenvalues of A is negative, differee real part of the eigenvalues of A is negative, differekkkkk- if the real part of the eigenvalues of A is negative, differential equation have stability solution 
- if the norm of eigenvalues less than 1， the powes of A go to 0

#### summary 
- if we have A n order differential equation, we will get a n by n matrix 