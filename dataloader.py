import random
import numpy as np

from utils import utils

class Word2Vocab():
    def __init__(self, dataset):
        
        self.dataset = dataset
        self.vocab_map = {}

        self.init_vocab()

    def init_vocab(self):
        for name, _ in self.dataset:
            for n in name:
                if not n in self.vocab_map:
                    self.vocab_map[n] = len(self.vocab_map) + 1

    def encode(self, name):
        return [self.vocab_map[n] for n in name]

    
class DataLoader():
    def __init__(self, dataset, vocab=None, shuffle=False, batch_size=1, max_len=0):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_len = max_len
        
        if shuffle:
            random.shuffle(dataset)

        if batch_size > len(dataset):
            raise ValueError('批次数量不能大于数据数量')

        split_size = len(dataset) / batch_size
        dataset_list = np.array_split(dataset, split_size)
        
        # 词表
        self.vocab = vocab

        # 编码 
        self.dataset = self.encode(dataset_list)

        self.current_index = -1
        self.max_index = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __iter__(self):
        return self 

    def __next__(self):
        self.current_index += 1
        if self.current_index >= self.max_index:
            return np.array([]) 

        return self.dataset[self.current_index]

    def is_next(self):
        return self.current_index < self.max_index 

    def encode(self, batch_dataset):
        convert_result = []
        for row in batch_dataset:

            name_sex_list = []
            for name, sex in row:
                name_codes = self.vocab.encode(name)

                if len(name_codes) < self.max_len:
                    name_codes = name_codes + [0] * (self.max_len - len(name_codes))

                name_sex_list.append((name_codes, [0, 1 if sex == '男' else 0, 1 if sex == '女' else 0]))  # （姓名列表，性别列表=>[男，女]）
            
            convert_result.append(name_sex_list)

        return convert_result
            



    
            
