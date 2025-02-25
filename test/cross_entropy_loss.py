import numpy as np
import torch

from torch.nn import CrossEntropyLoss


def cross_entropy_loss_by_my(logits, target):
    predict_distribution = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    print(predict_distribution)

    print(np.exp(logits))
    print(np.sum(np.exp(logits), axis=-1, keepdims=True))
    print(np.sum(np.exp(logits)))
    
    target_distribution = np.array([[1, 0, 0], [0, 0, 1]])

    loss_val = -np.mean(np.sum(target_distribution * np.log(predict_distribution), axis=1))
    print(loss_val)

def cross_entropy_loss_by_new_1(logits, target):
    sample_mean = np.sum(np.exp(logits), axis=-1, keepdims=True)
    print(sample_mean)
    print(np.log(sample_mean))

    target_distribution = np.array([[1, 0, 0], [0, 0, 1]]) == 1
    print(logits[target_distribution])

    loss_val = np.mean(-logits[target_distribution] + np.log(sample_mean))

    print(loss_val)

    
def cross_entropy_loss_by_pytorch(logits, target):
    loss_fn = CrossEntropyLoss(reduction='mean')
    loss_val= loss_fn(torch.tensor(logits), torch.tensor(target))

    print(loss_val.numpy())

if __name__ == '__main__':
    
    # 有3种类别，输入两个样本，第一个样本类别的索引为 0，第二个样本类别的索引为 2
    
    # 预测结果
    logits = np.array([[0.0541, 0.1762, 0.9489], [-0.0288, -0.8072, 0.4909]])

    # 真实结果
    target = np.array([0, 2])
    
    # 使用自己的交叉熵损失
    #cross_entropy_loss_by_my(logits, target)
    #print()

    # 使用 pytorch 的交叉熵损失
    cross_entropy_loss_by_pytorch(logits, target)
    print()

    cross_entropy_loss_by_new_1(logits, target)


