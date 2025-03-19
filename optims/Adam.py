import numpy as np

class Adam():
    def __init__(self, model, lr=1e-2, betas=(0.9, 0.999), eps=1e-08):
        self.model = model
        self.learning_rate = lr
        
        self.time = 0
        self.betas = betas
        self.epsilon = eps  # 稳定器，若没有此参数，np.sqrt() 的结果有可能等于 0 ，从而导致梯度消失。
        
        layer_size = len(self.model.layers)
        self.momentum = {'weight': [0] * layer_size, 'gamma': [0] * layer_size, 'beta': [0] * layer_size}
        self.velocity = {'weight': [0] * layer_size, 'gamma': [0] * layer_size, 'beta': [0] * layer_size}
    
    def step(self):
        layer_items = list(self.model.layers.values())

        for i in range(len(layer_items)):
            linear_layer, dropout_layer, normal_layer = layer_items[i]['linear'], layer_items[i]['dropout'], layer_items[i]['normal']
            gradient, gamma_gradient, beta_gradient = layer_items[i]['gradient'], layer_items[i]['gamma_gradient'], layer_items[i]['beta_gradient']

            # 更新权重参数 
            new_weight = self.caclulate_weight('weight', i, gradient, linear_layer.weight_matrix)
            #print(f'第{i}层的新权重:', new_weight)
            linear_layer.update_weight(new_weight)
            
            if normal_layer != None:
                # 计算当前层归一化缩放因子参数, 注意需要按照特征维度，合并多个样本的值
                new_gamma = self.caclulate_weight('gamma', i, gamma_gradient, normal_layer.gamma)
                normal_layer.update_gamma(new_gamma)
                
                # 计算当前层平移参数
                new_beta = self.caclulate_weight('beta', i, beta_gradient, normal_layer.beta)
                normal_layer.update_beta(new_beta)


        self.time += 1

    def caclulate_weight(self, _type, layer_number, gradient, weight):
        """
        计算新权重
        -param _type str 需要计算哪个类别的权重
        -param layer_number 层编号
        -param gradient tensors 当前的梯度信息
        -param weight tensors 当前的权重信息
        """
        self.momentum[_type][layer_number] = self.betas[0] * self.momentum[_type][layer_number] + (1 - self.betas[0]) * gradient
        self.velocity[_type][layer_number] = self.betas[1] * self.velocity[_type][layer_number] + (1 - self.betas[1]) * gradient ** 2
        #print('1,', self.momentum[i])
        #print('2,', self.velocity[i])
        
        # 偏差矫正（补偿初始零偏置）
        momentum = self.momentum[_type][layer_number] / (1 - self.betas[0] ** self.time)
        velocity = self.velocity[_type][layer_number] / (1 - self.betas[1] ** self.time)

        #print(momentum)
        #print(velocity)
        #print()
        return np.array(weight) - (self.learning_rate / np.sqrt(velocity + self.epsilon)) * momentum

