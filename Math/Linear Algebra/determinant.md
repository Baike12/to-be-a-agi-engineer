one reason : for eigen values

$$
\lvert \mathbf{A} \rvert = \det(\mathbf{A})
$$

- matrix is invertible indicate determinant is not 0
### properties:
- det(I)=1
- exchange rows reverse the sign of determinant 
- t times one row of matrix, the determinant is t times, **determinant is linear for each one row**
- if two rows equal, determinant is 0
- subtract i times row i from row j, **determinant do not change** 
- if one row is zero, determinant is zero
- the determinant of a triangle matrix is the product of diagonals
- when A is singular, det A is 0
- det(AB)=det(A)det(B), det(A inverse)=1/det(A)
- A transpose determinant equal A determinant 
#####  exchange rows reverse the sign of determinant 
$$
\begin{vmatrix}
a & b  \\
c  & d
\end{vmatrix}=ad-bc\to \begin{vmatrix}
c & d \\
a & b
\end{vmatrix}=bc-ad
$$
##### t times one row of matrix, the determinant is t times, **determinant is linear for each one row**
$$
\begin{vmatrix}
ta & tb \\
tc & td
\end{vmatrix}=t\begin{vmatrix}
a & b \\
c & d
\end{vmatrix}
$$
$$
\begin{vmatrix}
a+a_{1} & b+b_{1} \\
c & d
\end{vmatrix}=\begin{vmatrix}
a & b  \\
c & d
\end{vmatrix}+\begin{vmatrix}
a_{1} & b_{1} \\
c & d
\end{vmatrix}
$$

#####  if two rows equal, determinant is 0
$$
\begin{vmatrix}
a & b \\
 a & b
\end{vmatrix}=0
$$

##### subtract i times row i from row j, **determinant do not change** 
$$
\begin{vmatrix}
a & b  \\
c -la & d-lb
\end{vmatrix}=\begin{vmatrix}
a & b  \\
c & d
\end{vmatrix}+\begin{vmatrix}
a & b  \\
-la & -lb
\end{vmatrix}
$$

##### if one row is zero, determinant is zero
$$
\begin{vmatrix}
0 & 0 \\
c & d
\end{vmatrix}=0\begin{vmatrix}
a & b \\
c & d
\end{vmatrix}=0
$$
##### the determinant of a triangle matrix is the product of diagonals
$$
\begin{vmatrix}
d_{1} & x_{1} & x_{2} \\
0 & d_{2} & x_{3} \\
0 & 0 & d_{3}
\end{vmatrix}
$$
x is any number,
m multiply row3 can make column 3 in row1 and row2 equal 0, finally get
$$
\begin{vmatrix}
d_{1} & 0 & 0 \\
0 & d_{2} & 0 \\
0 & 0 & d_{3}
\end{vmatrix}=d_{3}d_{2}d_{1}\begin{vmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{vmatrix}
$$

##### when A is singular, det A is 0
- eliminate A to U, there is at least one row is zero 

#####  det(AB)=det(A)det(B), det(A inverse)=1/det(A)
$$
\begin{align}
\det\mathbf{A}\mathbf{B} & =\det\mathbf{A}\det\mathbf{B}\\  
\mathbf{A}^{-1}\mathbf{A} &= \mathbf{I} \\
\det \mathbf{A}^{-1}\mathbf{A} &= \det \mathbf{A}^{-1}\det \mathbf{A} \\
 1 ~~~ &= \det \mathbf{A}^{-1}\det \mathbf{A} \\
\det \mathbf{A}^{-1} &= \frac{1}{\det \mathbf{A}^{-1}}
\end{align}
$$
and
$$
\det \mathbf{A}^{2}=(\det \mathbf{A})^{2}
$$
$$
\det 2\mathbf{A}=2^{n}\det \mathbf{A}
$$

#####  A transpose determinant equal A determinant 
$$
\det \mathbf{A}^{T}=\det \mathbf{A}
$$
prove
$$
\mathbf{A}=\mathbf{L}\mathbf{U}
$$
- L is a low triangle matrix which all of diagonal is 0

so 

$$
\begin{align}
\det \mathbf{L} &= 1 \\
\end{align}
$$
and
$$
\begin{align} \\
\det \mathbf{A} &= \det \mathbf{L}\det \mathbf{U} \\
\det \mathbf{A}^{T} &= \det \mathbf{U}^{T}\det \mathbf{L}^{T} \\
 &= \det \mathbf{U}^{T} \\
 &= d_{1}\times d_{2}\dots ~~~ of ~~~ \mathbf{U} \\
 &= \det \mathbf{U} \\
 &= \det \mathbf{A}
\end{align}
$$
### formula for det A
$$
\begin{align}
\begin{vmatrix}
a & b \\
c & d
\end{vmatrix} &= \begin{vmatrix}
a & 0 \\
c & d
\end{vmatrix}+\begin{vmatrix}
0 & b \\
c & d
\end{vmatrix} \\
 &= \begin{vmatrix}
a & 0 \\
c & 0
\end{vmatrix}+\begin{vmatrix}
a & 0 \\
0 & d
\end{vmatrix}+\begin{vmatrix}
0 & b \\
c & 0
\end{vmatrix}+\begin{vmatrix}
0 & b \\
0 & d
\end{vmatrix} \\
 &= 0+ad-bc+0 \\
 &= ad-bc
\end{align}
$$
$$
\begin{align}
\det \mathbf{A} & =\sum_{i=1}^{N}a_{1i}a_{2j}a_{3k}\dots a_{ns}, ~~~ i,j,k\dots s ~~~ \in(1,2,3\dots n) \\ 
 & n! ~~~   items
\end{align}
$$
#### cofactor
choose one element  1 aij, cofactor is the determinant which do not contain row i and column j.

$$
cofactor ~~~ a_{ij}=C_{ij}
$$

the sign of C:
- plus:if i+j is even
- minus : if i+j is odd


$$
\det \mathbf{A}=a_{11}C_{11}+a_{12}C_{12}\dots
$$

**best way to get determinant is elimation to triangle matrix **

### application of determinant 
#### inverse
$$
\mathbf{A}^{-1}=
\begin{bmatrix}
a & b  \\
c & d
\end{bmatrix}^{-1}=\frac{1}{ad-bc}\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}=\frac{1}{\det \mathbf{A}}\begin{bmatrix}
C_{11} & C_{21} \\
C_{12} & C_{22}
\end{bmatrix}
$$
$$
\mathbf{A}^{-1}=\frac{1}{\det \mathbf{A}}\mathbf{C}^{T}
$$
- detA is product of n entries 
- C transpose is product of n-1 entries 

prove:

$$
\begin{align}
\mathbf{A}^{-1} &= \frac{1}{\det \mathbf{A}}\mathbf{C}^{T} \\
\mathbf{I} &= \frac{\mathbf{A}}{\det \mathbf{A}}\mathbf{C}^{T} \\
\mathbf{A}\mathbf{C}^{T} &= \det \mathbf{A}\mathbf{I} \\
\begin{bmatrix}
a_{11} & \dots & a_{1n} \\
\dots \\
a_{n1} & \dots & a_{nn}
\end{bmatrix}\begin{bmatrix}
c_{11} & \dots & c_{n1} \\
\dots \\
c_{1n} & \dots & c_{nn}
\end{bmatrix} &= \begin{bmatrix}
\det \mathbf{A} & \dots & 0 \\
0 & \det \mathbf{A} & \dots  \\
0 & \dots & \det \mathbf{A}
\end{bmatrix}
\end{align}
$$
- there is a rule: **row i multiply the adjoint matrix of other row equal 0**

$$
\begin{bmatrix}
a_{i_{1}} & \dots & a_{in}
\end{bmatrix}\begin{bmatrix}
c_{j1} \\
\dots \\
c_{jn}
\end{bmatrix}=\begin{cases}
\det \mathbf{A}, & i = j \\
0, & i  \neq j
\end{cases}
$$

##### summary :
- we have A, want A inverse 
- calculate the determinant of A 
- calculate the cafactor of every element , get C, the get C transpose 
- finally get A inverse 

#### cramer rule

$$
\begin{align}
\mathbf{A}\vec{x} &= \vec{b} \\
\vec{x} &= \mathbf{A}^{-1}\vec{b}  \\
 &= \frac{1}{\det \mathbf{A}}\mathbf{C}^{T}\vec{b} \\
 x_{1} &= \frac{\det \mathbf{B}_{1}}{\det \mathbf{A}}
\end{align}
$$

and B1 is A with column 1 replace by b


$$
\mathbf{B} _{1}=\begin{bmatrix}
\vec{b} &| n-1 & colums ~~~ of ~~~ \mathbf{A}
\end{bmatrix}
$$

detB:


$$
\det \mathbf{B_{1}}=b_{1}C_{11}+b_{2}C_{21}\dots b_{n}C_{n1}
$$



##### summary :
- calculate every determinant that column i replace by b

again:  **best way to get determinant is elimation to triangle matrix **

#### |detA| = volume of a box
the **row vector** compose the box
the sign of detA:
- indicate left-handed box or right-handed box

orthogonal matrix is a cube that turned in space 

prove Q determinant is 1 or -1:



$$
\begin{align}
because ~~~ \mathbf{Q}^{T}\mathbf{Q} &= \mathbf{I} \\
\begin{vmatrix}
\mathbf{Q}^{T}\mathbf{Q} 
\end{vmatrix} &= \begin{vmatrix}
\mathbf{I}
\end{vmatrix} \\
\begin{vmatrix}
\mathbf{Q}^{T}
\end{vmatrix} 
\begin{vmatrix} 
\mathbf{Q}
\end{vmatrix} &= 1 \\
\begin{vmatrix}
\mathbf{Q}
\end{vmatrix}^{2} &= 1
\end{align}
$$

so Q determinant is 1 or -1

##### properties :
- the volume of I is 1
- exchange two row of matrix, volume do not change 
- if one row multiply t, the volume multiply t

##### in 2 dimension space it is area
we can get area of **parallelogram** only through coordinate
- there is two vertex (a,b),(c,d),the area is 

$$
ad-bc
$$

also, we can get the area of **triangle** if we know the coordinate of the side of triangle 

$$
\frac{1}{2}(ad-bc)
$$

- if the vertex does not at origin, we have 3 coordinates of vertex of triangle

$$
\begin{align}
(x_{1},y_{2}), ~~~ (x_{2},y_{2}), ~~~ (x_{3},y_{3}) \\
area &= \begin{vmatrix}
x_{1} & y_{1} & 1 \\
x_{2} & y_{2} & 1 \\
x_{3} & y_{3} & 1
\end{vmatrix}
\end{align}
$$

