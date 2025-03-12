import numpy as np

class NormalLayer():
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        self.net_input = None
        self.net_input_normal = None
        
        self.normalized_shape = normalized_shape

        self.gamma = np.ones(self.normalized_shape)  # 缩放因子
        self.beta = np.zeros(self.normalized_shape)  # 偏移因子

    def __call__(self, net_input):
        # 归一化
        self.net_input_normal = self.standardization(net_input)
        #print('归一化净输入:', self.net_input_normal)
        #print('归一化均值', np.mean(self.net_input_normal, axis=1))
        #print('归一化标准差', np.var(self.net_input_normal, axis=1))
        #print()

        #self.net_input_normal = self.rescaling(self.net_input)
        #print('归一化净输入:', self.net_input_normal)

        # 二次仿射变换
        self.net_input = self.affine_fn_by_normal(self.net_input_normal)
        #print('二次仿射变换:', self.net_input)
        
        return self.net_input

    def standardization(self, z):
        """
        标准归一化净输入的值
        -param z tensors 净输入
        return tensors
        """
        # 注意，这里是层归一化处理方法，因此需要对每个样本进行求值，而非 np.mean(z)，当作是mini-batch的样本
        mean_value = np.mean(z, axis=1, keepdims=True) # 均值
        std_value = np.std(z, axis=1, keepdims=True)   # 标准差
        #print('mean_value', mean_value)
        #print('std_value', std_value)

        return (z - mean_value) / (std_value + 1e-5)

    def affine_fn_by_normal(self, z):
        """
        层归一化后的仿射变换
        -param z tensor 净输入
        return tensors
        """
        return self.gamma * z + self.beta
