import numpy as np

"""平方根传播"""
class RMSProp():
    def __init__(self, model, lr=0.01, alpha=0.99, eps=1e-8):
        self.model = model
        self.learning_rate = lr

        self.alpha = alpha
        self.epsilon = eps 
        
        layer_size = len(self.model.layers)
        self.ema = {'weight': [0] * layer_size, 'gamma': [0] * layer_size, 'beta': [0] * layer_size}

    def step(self):
        layer_items = list(self.model.layers.values())

        for i in range(len(layer_items)):
            linear_layer, dropout_layer, normal_layer = layer_items[i]['linear'], layer_items[i]['dropout'], layer_items[i]['normal']
            gradient, gamma_gradient, beta_gradient = layer_items[i]['gradient'], layer_items[i]['gamma_gradient'], layer_items[i]['beta_gradient']

            new_weight = self._algorithm('weight', i, gradient, linear_layer.weight_matrix)
            linear_layer.update_weight(new_weight)

            if normal_layer != None:
                # 计算当前层归一化缩放因子参数, 注意需要按照特征维度，合并多个样本的值
                new_gamma = self._algorithm('gamma', i, gamma_gradient, normal_layer.gamma)
                normal_layer.update_gamma(new_gamma)
                
                # 计算当前层平移参数
                new_beta = self._algorithm('beta', i, beta_gradient, normal_layer.beta)
                normal_layer.update_beta(new_beta)


    def _algorithm(self, _type, i, gradient, weight):
        """
        计算新权重
        -param _type str 需要计算哪个类别的权重
        -param layer_number 层编号
        -param gradient tensors 当前的梯度信息
        -param weight tensors 当前的权重信息
        """
        self.ema[_type][i] = self.alpha * self.ema[_type][i] + (1 - self.alpha) * gradient ** 2
        return weight - self.learning_rate / (np.sqrt(self.ema[_type][i]) + self.epsilon) * gradient
