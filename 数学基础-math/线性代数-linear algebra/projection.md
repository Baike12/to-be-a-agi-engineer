### projection
project b onto a
$$\begin{align}
\vec{a}^{T}(\vec{b}-x\vec{a}) & =0 \\
 x\vec{a}^{T}\vec{a} & =\vec{a}^{T}\vec{b} \\
x & =\frac{\vec{a}^{T}\vec{b}}{\vec{a}^{T}\vec{a}} \\
\end{align}
$$
projection vector p is:
$$
\begin{align}

 & p=\vec{a}x  \\
 & p=\vec{a}\frac{\vec{a}^{T}\vec{b}}{\vec{a}^{T}\vec{a}}
\end{align}
$$
projection matrix is :
$$
\begin{align}
\mathbf{P}=\frac{\vec{a}\vec{a}^{T}}{\vec{a}^{T}\vec{a}}
\end{align}
$$
- rank of P is 1
	- a column vector multiply a row vector is a matrix which rank is 1
- is symmetric
- if projection twice,  also itself
	- 
$$
	  \mathbf{P}^{2}=\mathbf{P}
$$
the function of  column space is: Pb is always in the column space of A
	
#### high dimension projection
why projection:
- Ax=b may have no solution 
- change b to the vector in the column space of A which is most close to b called p, Ax = p
let:
$$
\mathbf{A}\hat{\vec{x}}=\vec{p}
$$
and
$$
\vec{e}=\vec{b}-\vec{p}
$$
e is perpendicular to the plane which consist with the column space of A
$$
\begin{align}
&  \vec{p}=c_{1}\vec{x}_{1}+c_{2}\vec{x}_{2}\dots  \\
& \vec{p}=\mathbf{A}\vec{x}
\end{align}
$$
so
$$
\vec{e}=\vec{b}-\mathbf{A}\vec{x}
$$
surpose, a1 and a2 is basis of A column space , a1 and a2 is perpendicular to e
$$
\begin{align}
\vec{a}_{1}^{T}(\vec{b}-\mathbf{A}\hat{\vec{x}})=0 \\
\vec{a}_{2}^{T}(\vec{b}-\mathbf{A}\vec{x})=0
\end{align}
$$
the matrix format:
$$
\begin{bmatrix}
\vec{a}_{1}^{T} \\
\vec{a}_{2}^{T} \\
\end{bmatrix}
(\vec{b}-\mathbf{A}\hat{\vec{x}})=\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$
$$
\mathbf{A}^{T}(\vec{b}-\mathbf{A}\hat{\vec{x}})=0
$$
**e is in null space of A transpose, and perpendicular to the column space of A**
$$
\begin{align}
\vec{e} ~~~ in ~~~ N(\mathbf{A}^{T}) \\
\vec{e} \perp C(\mathbf{A})
\end{align}
$$
$$
\begin{align}
\mathbf{A}^{T}\vec{b} & =\mathbf{A}^{T}\mathbf{A}\hat{\vec{x}} \\
\hat{\vec{x}} & =(\mathbf{A}^{T}\mathbf{A})^{-1}(\mathbf{A}^{T}\vec{b}) \\
 because ~~~ \vec{p} & =\mathbf{A}\hat{\vec{x}} \\
\vec{p} & =\mathbf{A}(\mathbf{A}^{T}\mathbf{A})^{-1}(\mathbf{A}^{T}\vec{b}) \\ 
\end{align}
$$
so we get a matrix multiply b is the prejection of b
$$
\begin{align}
\vec{p} & =\mathbf{A}(\mathbf{A}^{T}\mathbf{A})^{-1}\mathbf{A}^{T}\vec{b} \\ 
\mathbf{P} & =\mathbf{A}(\mathbf{A}^{T}\mathbf{A})^{-1}\mathbf{A}^{T}
\end{align}
$$
if A is a invertible square , P is I, because the column space of A is whole Rn 
- P transpose is P 
	- P inverse then transpose is P transpose then inverse. P is invertible 
$$
(\mathbf{P}^{-1})^{T}=(\mathbf{P}^{T})^{-1}
$$
- P squared is P
$$
\begin{align}
\mathbf{P}^{T}=P \\
\mathbf{P}^{2}=\mathbf{P}
\end{align}
$$
### usage:
if we want find a best line in a plane to fit dots in plane
- there are more than 3 dots 
- surpose the line is y=d+cx
we can bring the coordinate of dots into the function of line, get a matrix,
$$
\mathbf{A}\vec{x}=\vec{b}
$$
the number of equations is more than variables,  it is no solution ,but we can solve the matrix 
$$
\mathbf{A}^{T}\mathbf{A}\hat{\vec{x}}=\mathbf{A}^{T}\vec{b}
$$
error is :
$$
\vec{e}=\mathbf{A}\vec{x}-\vec{b}
$$
minimize error :
$$
\lVert \mathbf{A}\vec{x}-\vec{b} \rVert =e_{1}^{2}+e_{2}^{2}+\dots
$$
$$
\vec{e}
$$
find:
$$
\hat{\vec{x}}=\begin{bmatrix}
\hat{{c}} \\
\hat{{d}}
\end{bmatrix}
$$
let:
$$
\begin{align}
\mathbf{A}^{T}\mathbf{A}\hat{\vec{x}}=\mathbf{A}^{T}\vec{b} \\
(\mathbf{A}^{T}\mathbf{A})\hat{\vec{x}}=\mathbf{A}^{T}\vec{b}
\end{align}
$$
get **normal equations**:
$$
\begin{bmatrix}
\mathbf{A}^{T}\mathbf{A}  & | & \mathbf{A}^{T}\vec{b}
\end{bmatrix}
$$
- this equations is the error, we can get this normal  equations through taking the derivative of parameter c and d from error equations of least square 

projection vector p is perpendicular to error e, also perpendicular to all vector in column space 
$$
\vec{p}\perp \vec{e}
$$
at last:
$$
\mathbf{P}=\mathbf{A}\hat{\vec{x}}
$$
#### a example 
we have dots :
$$
\begin{align}
(1,1) \\
(2,1) \\
(3,2)
\end{align}
$$
we have line :
$$
y=c+dt
$$
get equations :
$$
\begin{align}
c+d=1 \\
c+2d=2 \\
c+3d=2
\end{align}
$$
get matrix :
$$
\begin{bmatrix}
1 & 1  \\
1 & 2   \\
1 & 3 
\end{bmatrix}
\begin{bmatrix}
c \\
d
\end{bmatrix}=
\begin{bmatrix}
1 \\
2 \\
2
\end{bmatrix}
$$
get normal equation :
$$
\begin{align} \\
\vec{x} & =\begin{bmatrix}
c \\
d
\end{bmatrix} \\

\mathbf{A}^{T}\mathbf{A}\hat{\vec{x}} & =\mathbf{A}^{T}\vec{b} \\
\begin{bmatrix}
1 &1 & 1 \\
1 & 2 & 3
\end{bmatrix}\times \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix}\times \begin{bmatrix}
c  \\
d
\end{bmatrix} & = 
\begin{bmatrix}
1 & 1 & 1  \\
1 & 2 & 3
\end{bmatrix}\times \begin{bmatrix}
1 \\
2 \\
2
\end{bmatrix} \\
\begin{bmatrix}
3 & 6 \\
6 & 14
\end{bmatrix}\times \begin{bmatrix}
c \\
d
\end{bmatrix} & =\begin{bmatrix}
5 \\
11
\end{bmatrix}
\end{align}
$$
solve normal equations :
$$
\begin{align}
c=\frac{2}{3} \\
d=\frac{1}{2}
\end{align}
$$
get the best line :
$$
y=\frac{2}{3}+\frac{1}{2}t
$$

#### A transpose multiply A
if A has independent columns, then A transpose A is invertible
- this is the necessary condition of least square 
##### prove A transpose A is invertible
equal to prove:
$$
\mathbf{A}^{T}\mathbf{A}\vec{x}=0
$$
have only zero solution
$$
\begin{align}
\mathbf{A}^{T}\mathbf{A}\vec{x} & =0 \\
\vec{x}^{T}\mathbf{A}^{T}\mathbf{A}\vec{x} & =\vec{x}^{T}0=0 \\
(\mathbf{A}\vec{x})^{T}\mathbf{A}\vec{x} & =0 \\
so ~~~ \mathbf{A}\vec{x} & = 0 \\
\end{align}
$$
because A has independent columns, so
$$
\vec{x} = 0
$$
get it!

#### orthogonal vector 




#### the usage of P: there is a Ax=b, Pb get the projection of b in the column space of A
- if b is in the column space of A(Ax=b) , Pb=b
$$
\begin{align}
\mathbf{P}\vec{b} & =\mathbf{A}(\mathbf{A}^{T}\mathbf{A})^{-1}\mathbf{A}^{T}\vec{b} \\
because ~~~ \vec{b} & =\mathbf{A}\vec{x} \\
so ~~~ \mathbf{P}\vec{b} & =\mathbf{A}(\mathbf{A}^{T}\mathbf{A})^{-1}\mathbf{A}^{T}\mathbf{A}\vec{x} \\
 & =\mathbf{A}\vec{x}
\end{align}
$$
- if b is perpendicular to the column space of A , Pb=0
	- **b is in the null space of A transpose** 
$$
\mathbf{A}^{T}\vec{b}=0
$$
so
$$
\begin{align}
 & \mathbf{P}\vec{b}=\mathbf{A}(\mathbf{A}^{T}\mathbf{A})^{-1}\mathbf{A}^{T}\vec{b} \\ 
 & \mathbf{P}\vec{b}=0
\end{align}
$$


**the column space of  A  is perpendicular to the null space of A transpose** 
- Pb: b project onto column space of A
- (I-P)b: b project onto null space of A transpose(N(Atr))
