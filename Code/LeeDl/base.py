import numpy as np
import matplotlib.pyplot as plt

def plot_function(func):
    def wrapper(x_range=(-10,10), num_points=1000, *args, **kwargs):
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = func(x, *args, **kwargs)
        plt.plot(x,y,label=func.__name__)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
    return wrapper

@plot_function
def sigmoid(x):
    return 1/(1+np.exp(-x))

@plot_function
def sigmoid_dericvative(x):
    s = 1/(1+np.exp(-x))
    return s*(1-s)

# def softmax(x):

def main():
    sigmoid_dericvative()

if __name__ == "__main__":
    main()




























