orthogonal vector q
$$
\vec{q}^{T}_{i}\vec{q}_{j} = \begin{cases}
0, ~~~ if ~~~ i\neq j \\
1, ~~~ if ~~~ i=j
\end{cases}
$$
#### standard orthogonal basis Q

$$
Q=\begin{bmatrix}
q_{1} & q_{2} & \dots
\end{bmatrix}
$$
and
- 
$$
\mathbf{Q}^{T}\mathbf{Q}=\mathbf{I}
$$
- the column vector in matrix is orthogonal to each other 
- the norm of every vector is 1
**Q can not be square**
and if Q is square:
$$
\mathbf{Q}^{T}=\mathbf{Q}^{-1}
$$
### orthogonal matrix make calculation simple 
#### example
$$
\mathbf{Q}=\frac{1}{3}\begin{bmatrix}
1 & -2  \\
2 & -1 \\
2 & 2
\end{bmatrix}
$$
#### projection matrix
$$
\begin{align}
\mathbf{P} & =\mathbf{Q}(\mathbf{Q}^{T}\mathbf{Q})^{-1}\mathbf{Q}^{T} \\
 because ~~~ \mathbf{Q}^{T}\mathbf{Q} & =\mathbf{I} \\
so ~~~ \mathbf{P} & =\mathbf{Q}\mathbf{Q}^{T}
\end{align}
$$
properties:
- if  matrix Q is square, P is I, because the column space of Q is full space, a vector project to full space need not to transform 
-  and PP=P, and P is symmetric matrix
$$
\begin{align}
\mathbf{P}\mathbf{P} & =(\mathbf{Q}\mathbf{Q}^{T})(\mathbf{Q}\mathbf{Q}^{T}) \\
 & =\mathbf{Q}\mathbf{Q}^{T}\mathbf{Q}\mathbf{Q}^{T} \\
 & =\mathbf{Q}(\mathbf{Q}^{T}\mathbf{Q})\mathbf{Q}^{T} \\
 & =\mathbf{Q}\mathbf{I}\mathbf{Q}^{T} \\
 & =\mathbf{Q}\mathbf{Q}^{T}=\mathbf{P}
\end{align}
$$
#### normal matrix 
$$
\begin{align}
\mathbf{Q}^{T}\mathbf{Q}\hat{\vec{x}} & =\mathbf{Q}^{T}\vec{b} \\
because ~~~ \mathbf{Q}^{T}\mathbf{Q} & =\mathbf{I} \\
\hat{\vec{x}} & =\mathbf{Q}^{T}\vec{b} \\
\hat{\vec{x}}_{i} & =\vec{q}^{T}_{i}\vec{b}
\end{align}
$$
### orthogonalization graham-schmidt
first, we want get orthogonal vector from any vector 
there are 2 vectors
$$
\vec{a},\vec{b}
$$
vector A is a, vector B is perpendicular to A
$$
\vec{B}\perp \vec{A}
$$
get B to satisfy the condition :
$$
\begin{align}
\vec{B} &=\vec{b} -\vec{p}  \\
&=  \vec{b}-\frac{\vec{A}^{T}\vec{b}}{\vec{A}^{T}\vec{A}}\vec{A}
\end{align}
$$
to prove B is perpendicular to A
$$
\begin{align}
\vec{A}^{T}\vec{B} & =\vec{A}^{T}\left( \vec{b}-\frac{\vec{A}^{T}\vec{b}}{\vec{A}^{T}\vec{A}}\vec{A} \right)\\ 
 &= 0
\end{align}
$$
we have vector c that independent to a and b
- c minus its projection in a direction 
- c minus its projection in b direction 
so
$$
\begin{align}
\vec{C}\perp \vec{A}\\ 
\vec{C}\perp \vec{B} \\
\end{align}
$$
$$
\vec{C}=\vec{c}-\frac{\vec{A}\vec{c}}{\vec{A}^{T}\vec{A}}\vec{A}-\frac{\vec{B}^{T}\vec{c}}{\vec{B}^{T}\vec{B}}\vec{B}
$$
normalize:
$$
\begin{align}
\vec{q}_{1} & =\frac{\vec{A}}{\lVert \vec{A} \rVert}\\ 
\vec{q}_{2} &= \frac{\vec{B}}{\lVert \vec{B} \rVert } \\
\vec{q}_{3} &= \frac{\vec{C}}{\lVert \vec{C} \rVert }
\end{align}
$$
 we have matrix A consist of vector A,B and C, matrix Q consist of q, **Q and A have the same column space ** 
 
 and exist a R, A=QR , and R is a triangle matrix
$$
\mathbf{Q}=\begin{bmatrix}
\vec{q}_{1} & \vec{q}_{2}
\end{bmatrix}, ~~~ \mathbf{A}=\begin{bmatrix}
\vec{a} & \vec{b}
\end{bmatrix}
$$
$$
\begin{align}
\mathbf{A} & =\mathbf{Q}\mathbf{R}\\ 
\mathbf{R} &= \mathbf{Q}^{-1}\mathbf{A} \\
 &= \mathbf{Q}^{T}\mathbf{A} \\
 &= \begin{bmatrix}
\vec{q}_{1}^{T}\vec{a} & \vec{q}_{1}^{T}\vec{b} \\
\vec{q}_{2}^{T}\vec{a} & \vec{q}_{2}^{T}\vec{b}
\end{bmatrix} \\
\vec{q}_{2}^{T}\vec{a} &= 0
\end{align}
$$