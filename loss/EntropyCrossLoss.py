
from loss.Loss import Loss

class EntropyCrossLoss(Loss):
    def __init__(self):
        super(EntropyCrossLoss, self).__init__()

        self.loss = 0.0
        self.loss_error = [0.0]
        self.batch_size = 0

    def __call__(self, predict, target):

