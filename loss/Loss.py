import numpy as np

class Loss():
    def __init__(self, model = None):
        self.model = model

        self.loss = 0.0
        self.loss_error = [0.0]
        self.batch_size = 0

    def __call__(self, predict, target):
        return self

    def item(self):
        return self.loss

    def calculate_layer_error(self, net_input, next_layer_error, next_layer_weight, delta_fn):
        """
        一般网络层的误差，是由下一层的“误差”和“权重”点积结果，与当前层激活函数对净输入的导数的逐元素积，得到的梯度
        -param net_input 当前网络层的净输入
        -param next_layer_error 下一层“误差”
        -param next_layer_weight 下一层“权重”
        """
        #print('上一层的梯度', np.dot(next_layer_error, next_layer_weight.T))
        return delta_fn(net_input) * np.dot(next_layer_error, next_layer_weight.T)

    def backward(self):
        """反向传播的计算方式是通过损失函数/权重，得到梯度值"""

        next_layer_error = 0.0
        next_layer_weight = []
        
        layer_items = list(self.model.layers.values())
         
        # 从后向前计算梯度
        for i in range(len(layer_items) - 1, -1, -1):
            linear_layer, dropout_layer = layer_items[i]['linear'], layer_items[i]['dropout']
            normal_layer, delta_fn = layer_items[i]['normal'], layer_items[i]['delta_fn']
            
            # 计算误差，这里误差指的既不是【上一层误差】也不是【当前层误差】，而是上一层与当前层连接的【变化率】
            if i == len(layer_items) - 1:
                # 计算【输出层】误差
                layer_error = self.loss_error
            else:
                # 计算【一般网络层】误差
                layer_error = self.calculate_layer_error(linear_layer.net_input, next_layer_error, next_layer_weight, delta_fn)
            
            #print(f'第{i}层误差', layer_error)

            # 获取上一层的输出结果, 若到了第一层，直接取输入值
            if i == 0:
                current_input = self.model.in_features
            else:
                current_input = linear_layer.input
            
            #print(f'第{i}层输入T', current_input.T)

            # 正则化处理
            if dropout_layer != None:
                layer_error *= dropout_layer.d

            # 计算当前层权重参数梯度：连接误差值 * 当前层输入（上一层的输出）, 由链式法则推导得出
            layer_items[i]['gradient'] = np.dot(current_input.T, layer_error) / self.batch_size
            
            if normal_layer != None:
                # 计算当前层归一化缩放因子参数, 注意需要按照特征维度，合并多个样本的值
                layer_items[i]['gamma_gradient'] = np.sum(normal_layer.net_input_normal * layer_error, axis=0) / self.batch_size
                
                # 计算当前层平移参数
                layer_items[i]['beta_gradient'] = np.sum(layer_error, axis=0) / self.batch_size
            
            # 记录当前信息，用于误差传播
            next_layer_error = layer_error
            next_layer_weight = np.array(linear_layer.weight_matrix) 

            #print('*'*80)

        
                
