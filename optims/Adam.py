import numpy as np

class Adam():
    def __init__(self, model, lr=1e-2, betas=(0.9, 0.999), eps=1e-08):
        self.model = model
        self.learning_rate = lr

        self.betas = betas
        self.epsilon = eps  # 稳定器，若没有此参数，np.sqrt() 的结果有可能等于 0 ，从而导致梯度消失。
        
        self.momentum = 0
        self.velocity = 0

    def step(self):
        layer_items = list(self.model.layers.values())

        # 初始化动量/速率
        if self.velocity == 0:
            self.velocity = [0] * len(layer_items)
            self.momentum = [0] * len(layer_items)
        
        for i in range(len(layer_items)):
            linear_layer, dropout_layer, normal_layer = layer_items[i]['linear'], layer_items[i]['dropout'], layer_items[i]['normal']
            gradient, gamma_gradient, beta_gradient = layer_items[i]['gradient'], layer_items[i]['gamma_gradient'], layer_items[i]['beta_gradient']

            self.momentum[i] = self.betas[0] * self.momentum[i] + (1 - self.betas[0]) * gradient
            self.velocity[i] = self.betas[1] * self.velocity[i] + (1 - self.betas[1]) * gradient ** 2
            #print('1,', self.momentum[i])
            #print('2,', self.velocity[i])

            # 偏差矫正（补偿初始零偏置）
            momentum = self.momentum[i] / (1 - self.betas[0])
            velocity = self.velocity[i] / (1 - self.betas[1])

            #print(momentum)
            #print(velocity)
            #print()
            new_weight = np.array(linear_layer.weight_matrix) - (self.learning_rate / np.sqrt(velocity + self.epsilon)) * momentum

            # 更新参数 
            #print(f'第{i}层的新权重:', new_weight)
            linear_layer.update_weight(new_weight)
            
            # 层归一化并不需要"动量"，批量归一化需要
            if normal_layer != None:
                # 计算当前层归一化缩放因子参数, 注意需要按照特征维度，合并多个样本的值
                new_gamma = normal_layer.gamma - self.learning_rate * gamma_gradient
                normal_layer.update_gamma(new_gamma)
                
                # 计算当前层平移参数
                new_beta = normal_layer.beta - self.learning_rate * beta_gradient
                normal_layer.update_beta(new_beta)
