import functional as Fn 

from models.Model import Model
from layers import LinearLayer, DropoutLayer, NormalLayer

class ClassifyModel(Model):
    def __init__(self, lr=0.0, seq_size=0, label_size=0):
        super(ClassifyModel, self).__init__()
     
        self.learning_rate = lr
        self.in_features = None
        
        # 若无必要，勿增实体。第0层没有神经元，就不要申明网络层次，让模型去学习他
        self.linear1 = LinearLayer(seq_size, 6)
        self.linear2 = LinearLayer(6, 6)
        self.linear3 = LinearLayer(6, label_size)

        self.dropout1 = DropoutLayer(0.8)
        self.dropout2 = DropoutLayer(0.5)

        self.normal1 = NormalLayer([6])
        self.normal2 = NormalLayer([6])

        self.activation_fn = Fn.ReLU

        self.layers = {
            'h1': {'linear': self.linear1, 'dropout': self.dropout1, 'normal': self.normal1, 'delta_fn': Fn.ReLU_delta},
            'h2': {'linear': self.linear2, 'dropout': self.dropout2, 'normal': self.normal2, 'delta_fn': Fn.ReLU_delta},
            'h3': {'linear': self.linear3, 'dropout': None, 'normal': None, 'delta_fn': None}
        }

    def forward(self, features, is_train=True):
        
        # 第0 层 输入层，没有神经元
        self.in_features = features
        #print('='*20, 'forward Start', '='*20)

        # 第一层
        h_1 = self.linear1(features)
        h_1 = self.activation_fn(h_1)

        h_1 = self.normal1(h_1)
        h_1 = self.dropout1(h_1)
        #print(f'h_1={h_1}')
        
        # 第二层
        h_2 = self.linear2(h_1)
        h_2 = self.activation_fn(h_2)

        h_2 = self.normal2(h_2)
        h_2 = self.dropout2(h_2)
        #print(f'h_2={h_2}')

        h_3 = self.linear3(h_2)
        #print(f'h_3={h_3}')
    
        #print('='*20, 'forward End', '='*20)
        
        if is_train:
            return h_3

        return Fn.arg_max(Fn.Sigmoid(h_3), axis=-1)

