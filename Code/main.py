def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def outer_func(x):
    def inner_func(y):
        return x+y
    return inner_func

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    a = outer_func(2)
    o=[a ,"a:"]; print(o[1], o[0])# a是一个返回的函数
    b = a(3)
    o=[b ,"b:"]; print(o[1], o[0])# outer_func已经被调用结束了，但是inner_func依然可以捕获
    
    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
