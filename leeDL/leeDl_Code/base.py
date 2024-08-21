import os.path

import numpy as np
import matplotlib.pyplot as plt
import threading


def plot_function(func):
    def wrapper(x_range=(-10,10), num_points=1000, *args, **kwargs):
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = func(x, *args, **kwargs)
        plt.plot(x,y,label=func.__name__)
    return wrapper

@plot_function
def sigmoid(x):
    return 1/(1+np.exp(-x))

@plot_function
def sigmoid_derivative(x):
    s = 1/(1+np.exp(-x))
    return s*(1-s)

def finalize_plot():
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# def softmax(x):

def show_figure(*funcs):
    for func in funcs:
        func()
    finalize_plot()

import torch
def my_concat():
    a = torch.randn(5,3)
    print(a)
    """
    保留前2项    
    """
    b = a[:2]
    print(b)
    """
    保留2项之后    
    """
    c = a[2:]
    print(c)
    """
    如果冒号右边是一个负数n，是删除最后n项    
    """
    d= a[:-3]
    print(d)

    x = torch.tensor([[1, 2],[3, 4],[5, 6],[7, 8], [9, 10]])
    x = x.repeat(1,3)
    print(x)
    x= x.view(5,3,2)
    print(x)
    x= x.view(5,3,2).permute(1,0,2)
    print(x)


def double_left_slash():
    print(11 // 2)

def dict1():
    dic = {}
    dic['1'] = [i for i in range (10)]
    print(dic)


import numpy as np
import torch
import torch.nn as nn

def con_2():
    # 定义输入图像 (3通道, 5x5 大小)
    input_image = torch.tensor([[[1, 2, 0, 1, 2],
                                 [2, 1, 0, 1, 2],
                                 [1, 2, 1, 2, 0],
                                 [0, 1, 2, 1, 0],
                                 [1, 2, 0, 1, 2]],

                                [[2, 1, 2, 1, 0],
                                 [1, 2, 0, 1, 2],
                                 [2, 1, 2, 0, 1],
                                 [1, 0, 1, 2, 1],
                                 [2, 1, 2, 1, 0]],

                                [[0, 1, 2, 1, 2],
                                 [2, 0, 1, 2, 1],
                                 [1, 2, 0, 1, 2],
                                 [2, 1, 2, 0, 1],
                                 [1, 0, 1, 2, 1]]], dtype=torch.float32)

    # 将输入图像添加批次维度，变成 (1, 3, 5, 5)
    input_image = input_image.unsqueeze(0)

    # 定义卷积层 (输入通道数3, 输出通道数1, 卷积核大小3x3)
    conv_layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0)

    # 初始化卷积核权重 (3个3x3的卷积核)
    conv_layer.weight = nn.Parameter(torch.tensor([[[[ 0,  1,  2],
                                                     [ 0,  1,  2],
                                                     [ 0,  1,  2]],

                                                    [[ 0, -1, -2],
                                                     [ 0, -1, -2],
                                                     [ 0, -1, -2]],

                                                    [[ 1,  0,  1],
                                                     [ 1,  0,  1],
                                                     [ 1,  0,  1]]]], dtype=torch.float32))

    # 假设偏置为0
    conv_layer.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    # 计算特征图
    output_feature_map = conv_layer(input_image)
    print(output_feature_map.shape)
    print(output_feature_map)

def test1():
    a = torch.arange(1,31)
    a= a.reshape(2,3,5)
    a=a.view(a.size()[0], -1)
    print(a.shape)

def test2():
    a = [1,2,3,4]
    b = [1,3,2,4]
    print((a==b))

def test_dict():
    a = [('a', 1)]
    a += [('b', 2)]
    print(a)
    b = dict(a)
    print(b)

def test_zip():
    batch = [
        (torch.tensor([[1, 2], [3, 4]]), 0, 'a'),  # 样本 1
        (torch.tensor([[5, 6], [7, 8], [9, 10]]), 1, 'b'),  # 样本 2
        (torch.tensor([[11, 12]]), 2, 'c')  # 样本 3
    ]
    a, b, c = zip(*batch)
    print(a)
    print(b)
    print(c)

def test_tesor():
    a = torch.tensor([0,1,2])
    print(a)
    b = torch.FloatTensor(a).long()
    print(b)



import sqlite3
def query():
    # 数据库文件是example.db，如果文件不存在，会自动在当前目录创建:
    conn = sqlite3.connect('C:\ProgramData\Cold Turkey\data-app.db')
    cursor = conn.cursor()

    # 获取所有表的名称
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = cursor.fetchall()

    # 遍历所有表并查询所有数据
    for table_name in tables:
        print(f"查询表 {table_name[0]} 的数据:")
        cursor.execute(f"SELECT * FROM {table_name[0]};")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        print("\n")  # 打印空行以分隔不同表的数据

    # 关闭Cursor和Connection:
    cursor.close()
    conn.close()

def strQ2B(ustring):
    """
    全角转半角
    """
    ss = []
    for s in ustring:
        print(s, "\t<-s")
        for uchar in s:
            print(uchar, "\t<-uchar")
            unicode_point = ord(uchar)
            print(unicode_point, "\t<-unicode_point")
            if unicode_point == 12288:
                unicode_point = 32
            elif (unicode_point >= 65281 and unicode_point <= 65374):   # 全角字符（除空格）根据关系转化，>65281并且<65374
                unicode_point -= 65248# 转换
            rstring = chr(unicode_point)
            print(rstring, "\t<-rstring")
        ss.append(rstring)
        aa = ''.join(ss)
        print(aa, "\t<-aa")
    return aa

def test_strq2b():
    sss = "as 12 rfs"
    # strQ2B(sss)
    a = 10
    o = [a ,"a:"]; print(o[1], o[0] )

def test_next():
    a = [1,2,3]
    b = iter(a)
    o = [next(b) ,"next(a):"]; print(o[1], o[0] )
    o = [next(b) ,"next(a):"]; print(o[1], o[0] )
    o=[next(b) ,"next(b):"]; print(o[1], o[0])

def test_matrix_add_ve():
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v = torch.tensor([10, 11, 12])
    o=[A ,"A:"]; print(o[1], o[0])
    o=[v ,"v:"]; print(o[1], o[0])
    b = v.t()
    o=[b ,"b:"]; print(o[1], o[0])
    
def test_minus1_to():
    x = torch.tensor(list(range(1,31)))
    o=[x ,"x:"]; print(o[1], o[0])
    x = x.reshape(5,6)
    o=[x ,"x:"]; print(o[1], o[0])
    o=[x[:,-1:] ,"x[:,-1:]:"]; print(o[1], o[0])
    
    
def test_smooth_loss():
    lprobs = torch.tensor(list(range(1,31)))
    o=[lprobs ,"lprobs:"]; print(o[1], o[0])
    lprobs= lprobs.reshape(5,6)
    o=[lprobs ,"lprobs:"]; print(o[1], o[0])

    target = torch.tensor(list(range(0,5)))
    o=[target ,"target:"]; print(o[1], o[0])

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)# 扩展标签，方便计算
        o=[target ,"target:"]; print(o[1], o[0])

    # nll: 负对数似然（Negative log likelihood），当目标是一个one-ho时的交叉熵。以下行与F.nll_loss相同
    nll_loss = -lprobs.gather(dim=-1, index=target)# 获取负对数似然损失
    o=[nll_loss ,"nll_loss:"]; print(o[1], o[0])

    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)#
    o=[smooth_loss ,"smooth_loss:"]; print(o[1], o[0])

    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)
    o=[nll_loss ,"nll_loss:"]; print(o[1], o[0])
    o=[smooth_loss ,"smooth_loss:"]; print(o[1], o[0])


def test_keep_dim():
    lprobs = torch.tensor(list(range(1,25)))
    o=[lprobs ,"lprobs:"]; print(o[1], o[0])
    lprobs =  lprobs.reshape(2,3,4)
    o=[lprobs ,"lprobs:"]; print(o[1], o[0])
    lprobs = lprobs.sum(dim=-1, keepdim=True)
    o=[lprobs ,"lprobs:"]; print(o[1], o[0])

def test_transposeconv():
    # 定义转置卷积层
    conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)

    # 输入张量，形状为 (batch_size, channels, height, width)
    input_tensor = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32)

    # 前向传播
    output_tensor = conv_transpose(input_tensor)

    print("Input Tensor Shape: ", input_tensor.shape)
    print("Output Tensor Shape: ", output_tensor.shape)
    print("Output Tensor: \n", output_tensor)

def test_transposeconv1():
    batch_size = 1
    feature_dim = 64
    input_tensor = torch.randn(batch_size, feature_dim * 8, 4, 4)

    # 定义转置卷积层
    conv_transpose = nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)

    # 前向传播
    output_tensor = conv_transpose(input_tensor)

    print("Input Tensor Shape: ", input_tensor.shape)
    print("Output Tensor Shape: ", output_tensor.shape)
    
def test_view1():
    x  =torch.tensor(list(range(1,31)))
    x = x.reshape(5,6)
    o=[x ,"x:"]; print(o[1], o[0])
    x = x.view(-1)
    o=[x ,"x:"]; print(o[1], o[0])

def test_randn():
    x = torch.randn(5,6)
    o=[x ,"x:"]; print(o[1], o[0])

def show_model(model_file):
    file = os.path.join('D:\ML\models', model_file)
    o=[file ,"file:"]; print(o[1], o[0])
    
    model = torch.load(file)
    model_state = model['state_dict']
    for key, value in model_state.item():
        o=[key ,"key:"]; print(o[1], o[0])
        o=[value ,"value:"]; print(o[1], o[0])



def main():
    query()


if __name__ == "__main__":
    main()


















