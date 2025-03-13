import numpy as np
import functional as Fn

from loss.Loss import Loss

class BCELoss(Loss):
    def __init__(self, model=None, reduction='mean'):
        super(BCELoss, self).__init__(model)

        self.reduction = reduction

    def __call__(self, predict, target):
        print(target)
        print(predict)

        self.batch_size = target.shape[0]
        
        # 损失值
        logist = Fn.Sigmoid(predict)
        loss = target * np.log(logist) + (1 - target) * np.log(1 - logist)
        
        if self.reduction == 'mean':
            self.loss = -np.mean(loss)
        else:
            self.loss = -np.sum(loss)
        
        print(logist)
        print(loss) 
        print()

        # 误差
        self.loss_error = predict - target

        return self
