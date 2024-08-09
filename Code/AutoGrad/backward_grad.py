from typing import List, NamedTuple, Callable, Dict, Optional
import numpy as np
_name = 1
def fresh_name():
    global _name
    name = f"v_{_name}"
    _name += 1
    return name

class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_name()

    def __repr__(self):
        return repr(self.value)

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
        dl_dx = dl_doutputs
        dx_dself = Variable(1.)
        dx_dother = Variable(1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        return [dl_dself, dl_dother]

    tape = Tape(input=[self.name, other.name], output=[x.name], propagate=propagate)
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
    tape = Tape(input=[self.name, other.name], output=[x.name], propagate=propagate)
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
    tape = Tape(input=[self.name], output=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x

def ops_log(self):
    x = Variable(np.log(self.value))
    print(f'{x.name} = log({self.name})')# 打印一下在对哪一个变量进行什么计算

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = Variable(1 / self.value)
        dl_dself = dl_dx * dx_dself
        return [dl_dself]

    # record the input and output of the op
    tape = Tape(input=[self.name], output=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x

class Tape(NamedTuple):
    input : List[str]
    output : List[str]
    propagate : 'Callable[List[Variable], List[Variable]]'# 类型注解，描述propagate是一个可调用的，参数为两个List的对象

gradient_tape : List[Tape] = []
def reset_tape():
    global _name
    _name = 1
    gradient_tape.clear()

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


reset_tape()

x = Variable.constant(2., name='v-1')
y = Variable.constant(5., name='v0')
f = Variable.log(x) + x*y -Variable.sin(y)
o=[f ,"f:"]; print(o[1], o[0])

dx, dy = grad(f, [x, y])
print("dx", dx)
print("dy", dy)