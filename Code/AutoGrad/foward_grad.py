import numpy as np

class ADTangent:
    def __init__(self, x, dx):
        self.x = x
        self.dx = dx

    def __str__(self):
        context = f'val:{self.x:4f}, grad:{self.dx:4f}'
        return context

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

    def log(self):
        x = np.log(self.x)
        dx = 1 / self.x * self.dx
        return ADTangent(x, dx)

    def sin(self):
        x = np.sin(self.x)
        dx = np.cos(self.x) * self.dx
        return ADTangent(x, dx)

x1 = ADTangent(x=2, dx=1)# 对x1求微分
x2 = ADTangent(x=5, dx=0)
f = ADTangent.log(x1) + x2*x1 - ADTangent.sin(x2)
o=[f ,"f:"]; print(o[1], o[0])


