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
 &= \det \mathbf{U}
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
\det \mathbf{A} & =\sum_{i=1}^{N} \\ 
 & n! ~~~   items
\end{align}
$$