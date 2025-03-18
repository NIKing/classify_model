import numpy as np
import functional as Fn

from loss.Loss import Loss

class BCELoss(Loss):
    def __init__(self, model=None, reduction='mean'):
        super(BCELoss, self).__init__(model)

        self.reduction = reduction

    def __call__(self, predict, target):
        #print('target=',target)
        #print('predict=',predict)

        self.batch_size = target.shape[0]
        
        # 预测概率分布
        logist = Fn.Sigmoid(predict)
        print('logist', logist)

        # 剪枝，需要把概率限制在一定范围内，否则下面的 1 - logist 会造成 log(0) 的情况
        logist = np.clip(logist, 1e-7, 1 - 1e-7)
        
        loss = target * np.log(logist) + (1 - target) * np.log(1 - logist)
        
        if self.reduction == 'mean':
            self.loss = -np.mean(loss)
        else:
            self.loss = -np.sum(loss)
        
        #print('logist',logist)
        #print(self.loss) 
        #print()

        # 误差, 注意交叉熵损失的误差，表示的是预测概率分布与目标概率分布的差异
        self.loss_error = logist - target

        return self
