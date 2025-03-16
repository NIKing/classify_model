import numpy as np

def ReLU(net_input):
    """ReLU激活函数，值域为max(0, z)"""
    return np.maximum(net_input, 0)

def ReLU_delta(_input):
    """ReLU导函数"""
    return np.where(_input, 1, 0)

def Leaky_relu(x, alpha=0.01):
    """Leaky-Relu激活函数"""
    return np.where(x > 0, x, x * alpha)

def Identical(net_input):
    """恒等激活函数"""
    return net_input

def Sigmoid(net_input):
    """损失函数-S型函数"""
    return 1 / (1 + np.exp(-net_input))

def Sigmoid_delta(net_input):
    return Sigmoid(net_input) * (1 - Sigmoid(net_input))

def SoftMax(net_input):
    """软最大化函数 - 转概率分布"""
    return np.exp(net_input) / np.sum(np.exp(net_input), axis=-1, keepdims=True)

def SoftMax_delta(net_input):
    pass

def arg_max(logits, axis=0):
    """取最大值"""
    return np.argmax(logits, axis) 
