#### 文章推荐：
Automatic Differentiation in Machine Learning: a Survey[[https://www.jmlr.org/papers/volume18/17-468/17-468.pdf]]

### 微分种类：
#### 符号微分
- 表达式膨胀
- 精确
#### 数值微分
- 差分求解
- 精度不准确
	- 截断误差
	- 舍入误差
- 复杂度高
	- 不适合ML
#### 自动微分
- 所有计算都由基本运算构成
- 基本运算的导数已知
- 链式法则求最重导数
- 使用表达式追踪获得前向传播中的计算

### 自动微分模式
#### 正向微分
- 假设有函数$f(x_{1}, x_{2}) = \ln(x_{1})+x_{1}x_{2}-\sin x_{2}$ 
- 令$x_{1}=2, x_{2}=5$ 
- 将计算过程分解为
	- $v_{1}=\ln x_{1}$ 
	- $v_{2}=x_{1}x_{2}$ 
	- $v_{3}=\sin x_{2}$ 
	- $v_{4}=v_{1}+v_{2}$ 
	- $v_{5}=v_{4}-v_{3}$ 
- 上述小步骤对$x_{1}$ 求导得到1，这时$x_{2}$ 是常量，其导数是0，也就是在每一个小步对自变量求导
	- $v_{1}'=\frac{1}{x_{1}}\times x_{1}'=\frac{1}{2}$ 
	- $v_{2}'=x_{2}\times x_{2}'+x_{1}\times x_{1}'=5$ 
	- $v_{3}'=\cos x_{2}'\times x_{2}'=\cos_{5}$ 
	- $v_{4}'=v_{1}'+v_{2}'=5.5$ 
	- $v_{5}'=v_{4}'-v_{3}'=5.5$ 
	- 得到f对$x_{1}$ 的导数
#### 反向微分
- 将输出当做y，并假设y=1输入当做x，比如上面的$v_{5}'=v_{4}'=v_{3}'$ 
- $\dot{v_{4}}=\dot{v_{5}}\frac{ \partial ^{} \dot{v5} }{ \partial \dot{v_{4}} ^{} }=1$ 
- $\dot{v_{3}}=\dot{v_{5}}\frac{ \partial ^{} \dot{v_{5}} }{ \partial \dot{v_{3}} ^{} }=-1$ 
#### 雅各比矩阵
- 对于一系列函数$f(x)=(f_{1}(x), f_{2}(x),\dots f_{m}(x))$ ，
- 其中$x=(x_{1}, x_{2}, \dots x_{n})$ 
- 要求$f(x)$ 对各个x的偏导
- 有雅各比矩阵定义
$$\begin{align}
J_{f}(x) &= \begin{bmatrix}
\frac{ \partial ^{} f_{1} }{ \partial x_{1} ^{} }  & \dots & \frac{ \partial ^{} f_{1} }{ \partial x_{n} ^{} }  \\
\dots \\
\frac{ \partial ^{} f_{m} }{ \partial x_{1} ^{} }  & \dots  & \frac{ \partial ^{} f_{m} }{ \partial x_{n} ^{} } 
\end{bmatrix}
\end{align}$$
- 行数等于输出分量数
- 列数等于输入分量数
##### 正向微分
-  假设$\vec{v}$ 是关于$l=g(\vec{y})$ 的梯度$\vec{v}=\left[ \frac{ \partial l }{ \partial y_{1} },\dots \frac{ \partial l }{ \partial y_{m} } \right]$ 
	- 也就是说l是y的函数
- 则l对$x_{1}$ 的梯度就是
$$\begin{align} 
\mathbf{J}\vec{v} &= \begin{bmatrix}
\frac{ \partial y_{1} }{ \partial x_{1} } \\
\dots \\
\frac{ \partial y_{m} }{ \partial x_{1} }
\end{bmatrix}
\end{align}$$
#####  反向微分
- l对$\vec{x}$的梯度是$\mathbf{J}^{T}\vec{v}$  
		- 这里的$\vec{v}$ 是所有的输入，正向的时候只是一个输入
#### 选择
- 输入数量 > 输出数量，使用反向模式，
	- 对一个输出$y_{i}$ 用一次反向模式，得到雅各比矩阵的一行
- 输出数量 > 输入数量，用正向模式
	- 对一个输入$x_{i}$ 用一次正向模式，得到雅各比矩阵一列
#### 优缺点
- 数值精度高、不会膨胀表达式
- 需要大量存储中间求导结果、占用内存多

### 自动微分实现
#### 操作符重载（pytorch实现方式）
- 重载基本表达式
- 运行时记录基本表达式和对应的组合关系
	- 使用链式求导法则对基本表达式的微分结果组合
#### 源码转换法（mindspore华为）


### 未来
#### 使用代码表达数学
#### 易用性
- 控制流表达问题
- 复杂数据类型表达：bool等
- 语言特性
#### 性能

### 前向微分代码实现
#### 定义前向微分类
##### 功能与原理
- 包含自变量x和对x求导后的值
- 重载__str__打印方便
##### 实现
```python
class ADTangent:
    
    # 自变量 x，对自变量进行求导得到的 dx
    def __init__(self, x, dx):
        self.x = x
        self.dx = dx
    
    # 重载 str 是为了方便打印的时候，看到输入的值和求导后的值
    def __str__(self):
        context = f'value:{self.x:.4f}, grad:{self.dx}'
        return context
```
#### 重载运算
##### 功能与原理
- 判断是否ADTangent，是就值和微分分别相加
	- 不是的话就是 + 一个常数，常数的导数为0，所以还是self的导数
	- 返回的时候用一个ADTangent包装
-  - 和 × 也是类似， × 特殊的是不是ADTangent的情况下导数要乘常数
##### 实现
```python
    def __add__(self, other):
        if isinstance(other, ADTangent):
            x = self.x + other.x
            dx = self.dx + other.dx
        elif isinstance(other, float):
            x = self.x + other
            dx = self.dx
        else:
            return NotImplementedError
        return ADTangent(x, dx)
        
    def __sub__(self, other):
        if isinstance(other, ADTangent):
            x = self.x - other.x
            dx = self.dx - other.dx
        elif isinstance(other, float):
            x = self.x - other
            dx = self.dx
        else:
            return NotImplementedError
        return ADTangent(x, dx)

    def __mul__(self, other):
        if isinstance(other, ADTangent):
            x = self.x * other.x
            dx = self.x * other.dx + self.dx * other.x
        elif isinstance(other, float):
            x = self.x * other
            dx = self.dx * other
        else:
            return NotImplementedError
        return ADTangent(x, dx)
```
#### 定义运算
##### 功能与原理
- 对于一个运算，要计算运算本身和运算的微分
##### 实现
```python
    def log(self):
        x = np.log(self.x)
        dx = 1 / self.x * self.dx
        return ADTangent(x, dx)

    def sin(self):
        x = np.sin(self.x)
        dx = self.dx * np.cos(self.x)
        return ADTangent(x, dx)
```
#### 测试
##### 功能与原理
- 假设有函数$f(x_{1},x_{2})=\ln(x_{1})+x_{1}x_{2}-\sin(x_{2})$  
- 求对$x_{1}$ 的微分
##### 实现
```python
x1 = ADTangent(x=2., dx=1)
x2 = ADTangent(x=5., dx=0)
f = ADTangent.log(x1) + x1 * x2 - ADTangent.sin(x2)
print(f)
```

### 反向微分代码实现
#### 定义一个记录计算过程名字的函数
##### 功能与原理
- 全局递增的变量名id可以区分计算步骤的名字
##### 实现
```python
from typing import List, NamedTuple, Callable, Dict, Optional

_name = 1
def fresh_name():
    global _name
    name = f'v{_name}'
    _name += 1
    return name
```

#### 定义Variable封装运算过程中的变量
##### 功能与原理
-  实例化
##### 实现
```python
class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_name()
    
    def __repr__(self):
        return repr(self.value)
    
    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes. 
    @staticmethod
    def constant(value, name=None):
        var = Variable(value, name)
        print(f'{var.name} = {value}')
        return var
    
    # Multiplication of a Variable, tracking gradients
    def __mul__(self, other):
        return ops_mul(self, other)
    
    def __add__(self, other):
        return ops_add(self, other)
    
    def __sub__(self, other):
        return ops_sub(self, other)
    
    def sin(self):
        return ops_sin(self)
    
    def log(self):
        return ops_log(self)
    
```

#### 定义Tape跟踪Variable的计算
##### 功能与原理
-  包含输入输出两个列表和一个可调用对象
##### 实现
```python
class Tape(NamedTuple):
    inputs : List[str]
    outputs : List[str]
    # apply chain rule
    propagate : 'Callable[List[Variable], List[Variable]]'
```

#### 定义全局Tape和重置Tape的方法
##### 功能与原理
-  
##### 实现
```python
gradient_tape : List[Tape] = []

# reset tape
def reset_tape():
    global _name
    _name = 1
    gradient_tape.clear()
```
#### 定义具体操作
##### 功能与原理
-  输入一个和self相乘的数
- 创建一个Variable变量用于输出
- 定义用于反向传播的propagate函数，这个函数是一个闭包，在后面使用的时候再调用
##### 实现
```python
def ops_mul(self, other):
	x = Variable(self.value*other.value)
	print(f'{x.name} = {self.name} * {other.name}')

	def propagate(dl_doutputs):# 用于反向传播的时候
		dl_dx = dl_doutputs# 损失对输出的导数
		dx_dself = other# 损失对当前数的导数等于乘进来的数，因为是相乘
		dx_dother = self# 损失对乘进来的数的导数
		dl_dself = dl_dx * dx_dself# 损失对当前数的导数德育损失的输出的偏导乘以输出对当前数的导数，这里是链式法则
		dl_dother = dl_dx * dx_dother
		dl_dinputs = [dl_dself, dl_dother]# 记录损失对输入的偏导
		return dl_dinputs

	tape = Tape(input=[self.name, other.name], output=[x.name],propagate=propagate)# 包装在tape中
	gradient_tape.append(tape)# 由一个全局微分变量记录
	return x
	
def ops_add(self, other):
    x = Variable(self.value + other.value)
    print(f'{x.name} = {self.name} + {other.name}')

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = Variable(1.)
        dx_dother = Variable(1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        return [dl_dself, dl_dother]
    
    # record the input and output of the op
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x

def ops_sub(self, other):
    x = Variable(self.value - other.value)
    print(f'{x.name} = {self.name} - {other.name}')

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = Variable(1.)
        dx_dother = Variable(-1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        return [dl_dself, dl_dother]
    
    # record the input and output of the op
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x

def ops_sin(self):
    x = Variable(np.sin(self.value))
    print(f'{x.name} = sin({self.name})')
    
    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = Variable(np.cos(self.value))
        dl_dself = dl_dx * dx_dself
        return [dl_dself]
    
    # record the input and output of the op
    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x

def ops_log(self):
    x = Variable(np.log(self.value))
    print(f'{x.name} = log({self.name})')
    
    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = Variable(1 / self.value)
        dl_dself = dl_dx * dx_dself
        return [dl_dself]
    
    # record the input and output of the op
    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x
```

#### 测试正向梯度
##### 功能与原理
-  
##### 实现
```python
reset_tape()

x = Variable.constant(2., name='v-1')
y = Variable.constant(5., name='v0')

f = Variable.log(x) + x * y - Variable.sin(y)
print(f)
```
#### 计算梯度
##### 功能与原理
-  
##### 实现
```python
def grad(l, results):
    dl_d = {}
    dl_d[l.name] = Variable(1.)
    o=[dl_d ,"dl_d:"]; print(o[1], o[0])

    def gather_grad(entries):
        return [dl_d[entry] if entry in dl_d else None for entry in entries]

    for entry in reversed(gradient_tape):
        o=[entry ,"entry:"]; print(o[1], o[0])
        dl_doutputs = gather_grad(entry.output)
        dl_dinputs = entry.propagate(dl_doutputs)

        for input, dl_dinputs in zip(entry.input, dl_dinputs):
            if input not in dl_d:
                dl_d[input] = dl_dinputs# 如果不存在对这个输入的梯度，就加上
            else:
                dl_d[input] += dl_dinputs# 存在则累加当前的梯度值

    for name, value in dl_d.items():
        print(f'd{l.name}_d{name} = {value.name}')

    return gather_grad(result.name for result  in results)
```
#### 测试梯度计算
##### 功能与原理
-  
##### 实现
```python
dx, dy = grad(f, [x, y])
print("dx", dx)
print("dy", dy)
```

#### 总结
- 在计算前向传播的时候使用闭包函数记录反向传播的计算过程和所需数据
- 计算输出对输入变量的梯度的时候使用闭包计算就行