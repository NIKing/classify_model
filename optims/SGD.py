import numpy as np

"""随机梯度下降"""
class SGD():
    def __init__(self, model, lr=1e-5, momentum=0):
        self.model = model
        self.learning_rate = lr
        
        self.momentum = momentum
        
        layer_size = len(self.model.layers)

        self.velocity = [0] * layer_size 
        self.gamma_velocity = [0] * layer_size
        self.beta_velocity = [0] * layer_size
        
    def zero_grad(self):
        pass

    def step(self):
        layer_items = list(self.model.layers.values())
        
        # 从后向前计算梯度
        for i in range(len(layer_items)):
            linear_layer, dropout_layer, normal_layer = layer_items[i]['linear'], layer_items[i]['dropout'], layer_items[i]['normal']
            gradient, gamma_gradient, beta_gradient = layer_items[i]['gradient'], layer_items[i]['gamma_gradient'], layer_items[i]['beta_gradient']
             
            #print(f'第{i}层的梯度:', gradient)
            
            # 新权重参数 - 添加"动量"
            self.velocity[i] = self.momentum * self.velocity[i] + gradient
            new_weight = np.array(linear_layer.weight_matrix) - self.learning_rate * self.velocity[i]

            # 更新参数 
            #print(f'第{i}层的新权重:', new_weight)
            linear_layer.update_weight(new_weight)
            
            if normal_layer != None:
                # 计算当前层归一化缩放因子参数, 注意需要按照特征维度，合并多个样本的值
                self.gamma_velocity[i] = self.momentum * self.gamma_velocity[i] + gamma_gradient
                new_gamma = normal_layer.gamma - self.learning_rate * self.gamma_velocity[i]
                
                normal_layer.update_gamma(new_gamma)
                
                # 计算当前层平移参数
                self.beta_velocity[i] = self.momentum * self.beta_velocity[i] + beta_gradient
                new_beta = normal_layer.beta - self.learning_rate * self.beta_velocity[i] 

                normal_layer.update_beta(new_beta)
            


            
