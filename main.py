import random
import numpy as np

from models import ClassifyModel 
from dataloader import DataLoader

from loss import BCELoss, CrossEntropyLoss
from feature import Features

seed = 40
random.seed(seed)
np.random.seed(seed)

# 效果最好的参数 [8e-3, 55],
id2label = {0: '男', 1: '女'}
id2tag = {0: 1, 1: -1}

model = ClassifyModel(lr=4e-4, seq_size=3, label_size=len(id2label))
loss_fn = BCELoss(model)

def train(train_dataset):
    
    # 迭代训练，用于查看损失函数变化
    for i in range(5):
        print('*'*40, f'第{i+1}轮次训练', '*'*40)

        model.train()
        
        train_data = DataLoader(train_dataset, shuffle=True, batch_size=10)
        batch_data = next(train_data)
         
        batch_num, loss_sum = 0, 0
        while train_data.is_next():
            features = [t.x for t in batch_data]
            results = [t.y for t in batch_data]

            feature_ids = np.array(features, dtype=np.int32)
            label_ids = np.array([[1 if r == 1 else 0, 1 if r == -1 else 0] for r in results], dtype=np.int32)

            # 模型预测
            outputs = model(feature_ids)
            
            # 手动计算损失函数 
            loss = loss_fn(outputs, label_ids)

            # 反向传播-计算梯度
            loss.backward()

            batch_num += 1
            loss_sum += loss.item()
            
            print(f'epoch:{i + 1}; batch_size:{batch_num}; loss:{loss_sum / batch_num}; ')
            print('')

            batch_data = next(train_data)


def test(test_dataset):
    model.eval()
    
    test_data = DataLoader(test_dataset, shuffle=False, batch_size=2)
    
    batch_data = next(test_data)

    result_list = []
    
    batch_num = 1
    while test_data.is_next():
        print(f'Test Epoech:{batch_num}/{len(test_dataset)}')
        
        features = [t.x for t in batch_data]
        results = [t.y for t in batch_data]
        #print(features)

        feature_ids = np.array(features, dtype=np.int32)
        outputs = model(feature_ids, is_train=False)
        
        #print(results)
        #print(outputs)
        #print()

        result_list.append(([id2tag[o] for o in outputs], results))

        batch_data = next(test_data)
        batch_num += 1

    p, r, f1 = evaluate(result_list)
    print(p, r, f1)

def evaluate(result_list):
    predict_total, gold_total, correct_total = 0, 0, 0
    for predict_list, gold_list in result_list:
        print(predict_list)
        print(gold_list)
        print()

        correct_total += len(list(filter(lambda x: x == True, [predict_list[i] == gold_list[i] for i in range(len(predict_list))])))
        predict_total += len(predict_list)
        gold_total += len(gold_list)
    
    print(predict_total, gold_total, correct_total)
    precise = correct_total / (predict_total + 1e-5)
    recall = correct_total / (gold_total + 1e-5)

    return precise, recall, 2 * precise * recall / (precise + recall)
        

if __name__ == '__main__':
    # 姓名分类器

    feature = Features()

    train_instance_list = feature.readInstance('./cnname/train.csv', 1000)
    train(train_instance_list)
    
    test_instance_list = feature.readInstance('./cnname/test.csv', 10)
    test(test_instance_list)


