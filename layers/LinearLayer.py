import math
import random

import numpy as np

class LinearLayer():
    def __init__(self, input_features=0, output_features=0, bias=True):

        # 神经元（输出维度）; 
        self.output_dim=output_features

        # 特征数量（输入维度）;
        self.input_dim=input_features
        
        self.bias=bias

        # 根据输入向量的数量(N)，初始化神经元矩阵(N * 6)，并进行 He 初始化
        # He初始化，对ReLU激活函数等正向激活情况做优化，使用N(0, 2/n)的分布, N 表示正太分布的意思；n 表示神经元数量
        #self.unit_matrix = [[random.random() * 2 / self.output_dim] * len(features) for i in range(self.input_dim)] 

        # 虽然，理论上权重矩阵的应该由【输入向量数（输入特征维度）* 神经元数量】组成
        # 但是，输入的向量不能在初始化的时候获取到（如果在执行的时候初始化，每执行一次训练都会重置权重），因此需要固定输入特征维度
        # 在bert模型中，hidden_size = 768, [batch_size, input_dim] * [input_dim, output_dim]
        self.weight_matrix = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(2. / self.input_dim)
        #print(self.input_dim, self.output_dim, self.weight_matrix.shape)
        

    def __call__(self, features):
        """这是代表神经元函数，每个输入都需要与权重参数发生线性变换，再经过非线性变换，最后输出"""
        #print('&'*30)
        self.input = features

        #print('输入:', features)
        #print('权重：', self.weight_matrix)
        #print('样本均值分布:', np.mean(features, axis=1))
        #print('样本方差分布:', np.var(features, axis=1))
        #print()

        # 仿射变换
        self.net_input = self.affine_fn(features)
        #print('净输入:', self.net_input)
        #print('净输入均值', np.mean(self.net_input, axis=1))
        #print('净输入标准差', np.std(self.net_input, axis=1))
        #print()

        return self.net_input
    
    def affine_fn(self, features):
        """仿射函数（当偏置等于 0 时，仿射函数就是线性函数 y = w*x ）"""
        return np.dot(features, self.weight_matrix)

    def update_weight(self, weights):
        """更新权重"""
        self.weight_matrix = weights


