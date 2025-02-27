from models.Model import Model
from layers import LinearLayer, DropoutLayer, NormalLayer

class ClassifyModel(Model):
    def __init__(self, lr=0.0, seq_size=0, label_size=0):
        super(ClassifyModel, self).__init__()
     
        self.learning_rate = lr
        self.in_features = None
        
        # 若无必要，勿增实体。第0层没有神经元，就不要申明网络层次，让模型去学习他
        #self.h0 = InputLayer(input_dim = 1, output_dim = 6)
        self.h1 = LinearLayer(input_dim = seq_size, output_dim = 6, is_normal = False)
        self.h2 = LinearLayer(input_dim = 6, output_dim = 6, is_normal = False)
        self.h3 = LinearLayer(input_dim = 6, output_dim = label_size, activation = 'SoftMax', is_normal = False)

        self.h1.dropout = DropoutLayer(0.8)
        self.h2.dropout = DropoutLayer(0.5)

        #self.h1.normal = NormalLayer()
        #self.h2.noraml = NormalLayer()

        self.layers = {
            #'h0': self.h0,
            'h1': self.h1,
            'h2': self.h2,
            'h3': self.h3
        }

    def forward(self, features):
        
        # 第0 层 输入层，没有神经元
        #h_0 = self.h0(features)
        
        self.in_features = features

        #print('='*20, 'forward Start', '='*20)

        # 第一层
        h_1 = self.h1(features)
        h_1 = self.h1.dropout(h_1)
        #print(f'h_1={h_1}')
        
        # 第二层
        h_2 = self.h2(h_1)
        h_2 = self.h2.dropout(h_2)
        #print(f'h_2={h_2}')

        h_3 = self.h3(h_2)
        #print(f'h_3={h_3}')
    
        #print('='*20, 'forward End', '='*20)

        return h_3

