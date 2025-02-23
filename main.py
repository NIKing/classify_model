import random
import numpy as np

from models import PointModel
from dataloader import DataLoader, Word2Vocab

from loss import CrossEntropyLoss

seed = 40
random.seed(seed)
np.random.seed(seed)

# 效果最好的参数 [8e-3, 55],
# [6e-3, 5]
label2id = {'未知': 0, '女': 2, '男': 1}
id2label = {0: '未知', 2: '女', 1: '男'}

model = PointModel(lr=1e-2, seq_size=3, label_size=len(label2id))
loss = CrossEntropyLoss(model)

def loss_callback(predict, target):
    return loss(predict, target)

def train(train_dataset):
    
    # 迭代训练，用于查看损失函数变化
    for i in range(45):
        
        print('*'*40, f'第{i+1}轮次训练', '*'*40)

        model.train()
        
        train_data = DataLoader(train_dataset, vocab=vocab, shuffle=True, batch_size=2, max_len=3)
        iter_data = iter(train_data)
        batch_data = next(iter_data)
         
        batch_num, loss_sum = 0, 0
        while train_data.is_next():
            features, results = zip(*batch_data)

            features = np.array(features, dtype=np.int32)
            results = np.array(results, dtype=np.int32)
            #print(features, features.shape)
            #print(results, results.shape)

            # 模型预测
            outputs = model(features)
            
            # 手动计算损失函数 
            loss = loss_callback(outputs, results)

            # 反向传播-计算梯度
            loss.backward()

            batch_num += 1
            loss_sum += loss.item()
            
            print(f'epoch:{i + 1}; batch_size:{batch_num}; loss:{loss_sum / batch_num}; ')
            print('')

            batch_data = next(iter_data)


def test(test_dataset):
    model.eval()
    
    test_data = DataLoader(test_dataset, vocab=vocab, shuffle=False, batch_size=1, max_len=3)
    
    iter_data = iter(test_data)
    batch_data = next(iter_data)
    
    i = 0
    while test_data.is_next():
        features, results = zip(*batch_data)
        
        features = np.array(features, dtype=np.int32)
        results = np.array(results, dtype=np.int32)
        #print(features, features.shape)
        #print(results, results.shape)
        
        print(f'Test Epoech:{i}/{len(test_dataset)}')
        output = model(features)
        print(output)
        output = np.argmax(output)

        print(output, id2label[output])
        print()

        batch_data = next(iter_data)
        i += 1


if __name__ == '__main__':
    # 姓名分类器
    train_dataset = [
        ['倪杰', '男'], ['郭家旭', '男'], ['刘亦菲', '女'], ['郭茜', '女'], ['李旭', '男'],
        ['周杰伦', '男'], ['谢如雪', '女'], ['王菲', '女'], ['王星', '女'], ['周星星', '男'],
        ['皮卡丘', '未知'], ['奥特曼', '未知']
    ]
    
    # 添加一些干扰
    #train_dataset = [[[s * 1e-6 for s in samples], target] for samples, target in train_dataset]

    #check_result = [2 * sum(sample) == target for sample, target in train_dataset]
    #print(check_result)

    test_dataset = [
        ['王杰', '男'], ['倪辉', '男'], ['周星弛', '男'], ['刘菲', '女']
    ]

    # 建立词表
    vocab = Word2Vocab(train_dataset + test_dataset)
    
    train(train_dataset)
    
    test(test_dataset)


