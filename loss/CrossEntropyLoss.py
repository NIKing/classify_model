import numpy as np
from loss.Loss import Loss

class CrossEntropyLoss(Loss):
    def __init__(self, model = None):
        super(CrossEntropyLoss, self).__init__(model)

        self.loss = 0.0
        self.loss_error = [0.0]
        self.batch_size = 0

    def __call__(self, predict, target):
        print('计算损失值')
        print(predict)
        print(target)
        print()
        
        pred_log = np.log(predict)
        
        classify_dot = np.dot(target, pred_log)

        loss = -np.sum(classify_dot, axis=-1, keepdims=True)

        print(pred_log)
        print(classify_dot)
        print(loss)

        print('计算结束')

        return self
