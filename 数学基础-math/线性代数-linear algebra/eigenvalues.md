### eigen vector 

$$
\mathbf{A}\vec{x}=\lambda\vec{x} 
$$

if A is singular, 0 is A eigen value a
#### for a projection matrix P(3d)

- vector in plane of P is a eigen vector, because Px=1x
- vector that is perpendicular to plane of P is a eigen vector , Px=0

#### for a permutation matrix 

$$
\mathbf{A}=\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
$$

and:

$$
\vec{x}_{1}=\begin{bmatrix}
1 \\
1
\end{bmatrix}
, ~~~ \mathbf{A}\vec{x}_{1}=\vec{x}
$$

and:


$$
\vec{x}_{2}=\begin{bmatrix}
 -1 \\
1
\end{bmatrix}, ~~~ \mathbf{A}\vec{x}_{2}=-\vec{x}_{2}
$$

#### properties :

- n by n matrix has n eigen values 
- the trace: sum of diagonal equal sum of eigen values 

#### calculate Ax=lambda x

$$
\begin{align}
\mathbf{A}\vec{x}&=\lambda \vec{x} \\
(\mathbf{A}-\lambda \mathbf{I})\vec{x} &= 0 \\
\end{align}
$$

(A-lambda x)is singular ,so 

$$
\det(\mathbf{A}-\lambda \mathbf{I})=0
$$

##### an example

$$
\mathbf{A_{1}}=\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}
$$

and:

$$
\det(\mathbf{A_{1}}-\lambda \mathbf{I})=
\begin{vmatrix}
3-\lambda & 1 \\
1 & 3-\lambda
\end{vmatrix}
=\lambda ^{2}-6\lambda+8
$$
cases:
- 6 is trace
- 8 is determinant 

$$
\begin{align}
\lambda_{1}&=2 \\
\lambda_{2} &= 4 \\
\vec{x}_{1} &= \begin{bmatrix}
1 \\
1
\end{bmatrix} \\
\vec{x}_{2} &= \begin{bmatrix}
1 \\
-1
\end{bmatrix}
\end{align}
$$

$$
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}=\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}+3\mathbf{I}
$$

- and eigen value of A1 is 3x of the eigen value of A
- eigen vector is the same

#### so, for I, but just for I

$$
\begin{align} \\
\mathbf{A}\vec{x} &= \lambda \vec{x} \\
(\mathbf{A}+3\mathbf{I})\vec{x}&= \mathbf{A}\vec{x}+3\mathbf{I}\vec{x}\\
 &= (\lambda+3)\vec{x}
\end{align}
$$

#### for rotation matrix Q

$$
\mathbf{Q}=\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

- trace = sum of eigen values = 0
- determinant = product of eigen values = 1
- damn! lambda1 = i, lambda2 = -i
the real matrix could have complex eigen value,but if the matrix is symmetric, that not happen, the eigen value of anti-symmetric matrix is only imaginary

#### the eign value of triangle matrix is the diagonal elements
example :

$$
\mathbf{A}=\begin{bmatrix}
3 & 1 \\
0 & 3
\end{bmatrix}
$$

$$
\begin{align}
\lambda_{1}&=3 \\
\lambda_{2} &= 3 \\
\end{align}
$$

get repeat eigen values
eiginvector is 

$$
\begin{align}
(\mathbf{A}-\lambda \mathbf{I})\vec{x}&=0 \\
 \begin{bmatrix}
0 & 1 \\
0 & 0
\end{bmatrix}\vec{x} &= 0 \\
\vec{x} &= \begin{bmatrix}
1 \\
0
\end{bmatrix}
\end{align}
$$

have only one eigen vector, this is a **degenerate matrix** ,occur only when repeat eigenvalues

### S: eigenvector matrix 
- S is invertible, S has n independent vectors
have:
$$
\begin{align}\\
\mathbf{A}\mathbf{S}=\mathbf{A}\begin{bmatrix}
\vec{x}_{1} & \vec{x}_{2} & \dots
\end{bmatrix} & =\begin{bmatrix}
\mathbf{A}\vec{x}_{1} & \mathbf{A}\vec{x}_{2} & \dots
\end{bmatrix}=\begin{bmatrix}
\vec{x}_{1} & \vec{x}_{2} & \dots
\end{bmatrix}\begin{bmatrix}
\lambda_{1} & 0 & \dots \\
0 & \lambda_{2} & \dots \\
0 & \dots & \lambda _{n}
\end{bmatrix}=S\Lambda \\
\mathbf{S}^{-1}\mathbf{A}\mathbf{S} &= \Lambda
\end{align}
$$
cases:
- Lambda is eigenvalues matrix 

$$
\mathbf{A}=\mathbf{S}\Lambda \mathbf{S}^{-1}
$$

#### eigenvalue of A square 

$$
\begin{align}
\mathbf{A}\vec{x}&=\lambda \vec{x} \\
\mathbf{A}\mathbf{A}\vec{x} &= \lambda \mathbf{A}\vec{x} \\
\mathbf{A}^{2}\vec{x} &= \lambda ^{2}\vec{x}
\end{align}
$$

the eigenvalues is square ,the eigenvector is the same

$$
\begin{align}
\mathbf{A}&=\mathbf{S}\Lambda \mathbf{S}^{-1} \\
\mathbf{A}^{2} &= \mathbf{S}\Lambda \mathbf{S}^{-1}\mathbf{S}\Lambda \mathbf{S}^{-1} \\
 &= \mathbf{S}\Lambda ^{2}\mathbf{S}
\end{align}
$$

the k-th power follow the same rule

$$
\mathbf{A}^{k}=\mathbf{S}\Lambda ^{k}\mathbf{S}
$$

the eigen values of the k-th power of matrix is the k-th power of  the eigenvalues of matrix 

#### stable matrix 
if eigenvalues below one, the matrix is stable 

$$
\lvert \lambda _{i} \rvert <1
$$

the base assumption is: **S has n independent vectors** 

#### A is sure to have n independent vectors, and be diagonalizable if A have n different eigenvalues 
if there is repeat:
- indentity matrix: have n eigenvalues and n enginvectors

if matrix A is a diagonal matrix 

$$
\Lambda=\mathbf{A}
$$

if matrix A is triangle matrix
- diagonal is eigenvalues
if A is 

$$
\mathbf{A}=\begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}
$$

the eigenvector and eigenvalues  is 

$$
\begin{align}
\lambda_{1}&=2 \\
\lambda_{2} &= 2 \\
\vec{x}_{1} &= \begin{bmatrix}
 1 \\
0
\end{bmatrix}
\end{align}
$$

have 2 repeat eigenvalues , so A is **not diagonalizable**

####

$$
u_{k+1}=\mathbf{A}u_{k} 
$$

assumption :

$$
u_{1} = \mathbf{A}u_{0}, u_{2} = \mathbf{A}u_{1}=\mathbf{A}^{2}u_{0},u_{100}=\mathbf{A}^{100}u_{0}
$$

get:

$$
\begin{align}
if ~~~ u_{0}&=\mathbf{S}\vec{c}=c_{1}\vec{x}_{1}+c_{2}\vec{x}_{2}+\dots  \\
u_{100} &= \mathbf{A}^{100}u_{0} \\
  &= \mathbf{A}^{100} c_{1}\vec{x}_{1}+\mathbf{A}^{100}c_{2}\vec{x}_{2}+\dots \\
 &= \lambda ^{100}c_{1}\vec{x}_{1}+\lambda ^{100}v_{1}\vec{x}_{1}+\dots \\
 &= \Lambda ^{100}\mathbf{S}\vec{c}
\end{align}
$$

x is eigenvectors 

the eigenvalues of symmetric matrix:
- real not complex 
- eigenvectors is orthogonal  

##### a example:fibnacci

$$
0,1,1,2,3,5,8\dots
$$

we can get the formula 

$$
\begin{align}
F_{k+2}&=F_{k+1}+F_{k} 
\end{align}
$$

define uk:

$$
u_{k}=\begin{bmatrix}
F_{k+1} \\
F_{k}
\end{bmatrix}
$$

expand formula to equation :

$$
\begin{align}
F_{k+2}&=F_{k+1}+F_{k}  \\
F_{k+1} &= F_{k+1} 
\end{align}
$$

get matrix :

$$
u_{k+1}=\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}\begin{bmatrix}
F_{k+1} \\
F_{k}
\end{bmatrix}=\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}u_{k}
$$

calculate eigenvalues and eigenvecotrs

$$
\begin{vmatrix}
1-\lambda & 1 \\
1 & -\lambda
\end{vmatrix}=\lambda ^{2}-\lambda-1 ~~~ similar  ~~~ F_{k+2}-F_{k+1}-F_{k}=0
$$

get 

$$
\begin{align}
\lambda_{1}&=\frac{{1+\sqrt{ 5 }}}{2} \\
\lambda_{1}&=\frac{{1-\sqrt{ 5 }}}{2} \\
\end{align}
$$

the bigger eigenvalue  determine how number increase in fibonacci, and we can infer :

$$
F_{100}=\frac{{1+\sqrt{ 5 }}}{2}^{100}c_{1}
$$

and find x1：

$$
\vec{x}_{1}=
\begin{bmatrix}
\lambda_{1} \\
1
\end{bmatrix},\vec{x}_{2}=\begin{bmatrix}
\lambda_{2} \\
1
\end{bmatrix}
$$

and:


$$
u_{0}=\begin{bmatrix}
F_{1} \\
F_{0}
\end{bmatrix}=\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

find c1 and c2

$$
c_{1}\vec{x}_{1}+c_{2}\vec{x}_{2}=u_{0}=\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

summary :
- for a first-order system,we know the relationship of its variables , starting vector is u0
- we transform its relationship to a matrix A
- the find the eigenvalues and eignvectors of A, and eigenvectors is different 
- then write u0 as combination of eigenvectors
- finally get k powers of A

$$
  \mathbf{A}^{k}=\mathbf{S}\Lambda ^{k}\mathbf{S}^{-1}
$$