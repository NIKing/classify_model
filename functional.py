import numpy as np

class Functional(): 
    @staticmethod
    def ReLU(net_input):
        """ReLU激活函数，值域为max(0, z)"""
        return np.maximum(net_input, 0)
    
    @staticmethod
    def Leaky_relu(x, alpha=0.01):
        """Leaky-Relu激活函数"""
        return np.where(x > 0, x, x * alpha)
    
    @staticmethod
    def ReLU_delta(_input):
        """ReLU导函数"""
        return np.where(_input, 1, 0)
    
    @staticmethod
    def Identical(net_input):
        """恒等函数"""
        return net_input

    @staticmethod
    def SoftMax(net_input):
        """软最大化函数"""
        return np.exp(net_input) / np.sum(np.exp(net_input))
    
    @staticmethod
    def SoftMax_delta(net_input):
        pass
