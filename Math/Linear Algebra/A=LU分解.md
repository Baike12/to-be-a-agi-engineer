### 矩阵A分解为LU，其中U是上三角矩阵
$$\displaylines{
\mathbf{E}\mathbf{A}=\mathbf{U}\\
\mathbf{E}^{-1}\mathbf{E}\mathbf{A}=\mathbf{E}^{-1}\mathbf{U}\\
\mathbf{A}=\mathbf{E}^{-1}\mathbf{U}\\
so ~~~ \mathbf{L}=\mathbf{E}^{-1}
}$$
L是下三角矩阵，且对角线都是1
还可以
$$\displaylines{
\mathbf{A}=\mathbf{L}\mathbf{D}\mathbf{U}
}$$
其中L是上三角矩阵，U是下三角矩阵，D是对角线矩阵
#### 对于三阶矩阵
$$\displaylines{
\mathbf{E}_{32}\mathbf{E}_{31}\mathbf{E}_{21}\mathbf{A}=\mathbf{U}\\
\mathbf{A}=\mathbf{E}_{32}^{-1}\mathbf{E}_{31}^{-1}\mathbf{E}_{21}^{-1}\mathbf{U}
}$$
E21是使A的第2行第1列为0的矩阵
#### 为什么使用A=LU而不是EA=U
因为L中相乘的顺序很好，L就是来源于每一个E逆的堆叠，如果是EA=U，得出E的过程并不是E32E31E21的简单堆叠，也就是说L可以简单体现每一个消元乘数，在上面的消元过程中不能存在行置换。
整个过程是这样的：
- 有一个矩阵nxn的A，要将它转换成上三角矩阵U，这个过程可以用左乘一些E实现
- 然后求这些E的逆，最后用这些E直接组成L
#### 一个nxn的矩阵消元要多少步：
假设一个元素变化是一次操作
$$\displaylines{
n^{2}+(n-1)^{2}+\dots+1\\
about ~~~ \frac{1}{3}n^{3}
}$$
因为
如果在方程组中，右侧的向量要计算
$$\displaylines{
n^{2}
}$$
### 行互换
主元位置为0时
左乘置换矩阵
置换矩阵：重新排列的单位矩阵
置换第一行和第二行：将单位阵的第一第二行互换
$$\displaylines{
\mathbf{P}_{12}=\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0  \\
0 & 0 & 1
\end{bmatrix}
}$$
##### 置换矩阵的逆等于其转置
$$\displaylines{
\mathbf{P}^{-1}=\mathbf{P}^{T}
}$$
$$\displaylines{
\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}
}$$
##### 有n的阶乘种置换矩阵
##### 置换矩阵都可逆
### 转置
$$\displaylines{
\mathbf{A}_{ij}=(\mathbf{A}^{T})_{ji}
}$$
##### 转置之后不变的就是对称矩阵
##### 矩阵的转置乘矩阵得到对称矩阵
$$\displaylines{
\mathbf{R}^{T}\mathbf{R} ~~~ is ~~~ symmetric ~~~
}$$
$$\displaylines{
(\mathbf{R}^{T}\mathbf{R})^{T}=\mathbf{R}^{T}\mathbf{R}^{T  T}=\mathbf{R}^{T}\mathbf{R}
}$$