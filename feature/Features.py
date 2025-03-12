from feature.common.TaskType import TaskType
from feature.tagset.TagSet import TagSet

from feature.featuremap.LockableFeatureMap import LockableFeatureMap

import pandas as pd

"""样本"""
class Instance():
    def __init__(self, x, y):
        """
        -param x 特征向量
        -param y 标签
        """
        self.x = x
        self.y = y

class Features():
    def __init__(self):

        # 特征映射
        self.featureMap = LockableFeatureMap(TagSet(TaskType.CLASSIFICATION))
        self.featureMap.mutable = True

    def readInstance(self, corpus):
        """
        从语料库读取实例, 在这里调用不同特征函数获取特征值
        -param corpus 语料库
        -param featureMap 特征映射
        return 数据集
        """
        instanceList = []
        lineIterator = pd.read_csv(corpus).loc[:,[True, True]].values
        #lineIterator = lineIterator[:2]
        for i, line in enumerate(lineIterator):
            print(f'训练条目总数:{i}/{len(lineIterator)}')
            text, label = line
            
            x = self.extractFeature(text, self.featureMap)
            y = self.featureMap.tagSet.add(label)

            if y == 0:
                y = -1
            elif y > 1:
                raise ValueError("类别数目大于2，目前只支持二分类")

            instanceList.append(Instance(x, y))

        return instanceList

    def extractFeature(self, text, featureMap):
        """特征提取"""
        featureList = []
        
        # 特征函数
        givenName = self.extractGivenName(text)

        # 特征模版
        self.addFeature("1" + givenName[:1], featureMap, featureList)
        self.addFeature("2" + givenName[1:], featureMap, featureList)
        self.addFeature("3" + givenName, featureMap, featureList)

        return featureList

    def extractGivenName(self, name) -> str:
        """
        提取姓名中的名字，去掉姓
        -param name 姓名
        return 名
        """
        if len(name) <= 2:
            return "_" + name[len(name) - 1:]

        return name[len(name) - 2:]

    def addFeature(self, feature, featureMap, featureList):
        """
        向特征向量插入特征
        -param str  feature 特征
        -param map  featureMap 特征映射, 由双数组树构成
        -param list featureList 特征向量
        """
        featureId = featureMap.idOf(feature)
        if featureId != -1:
            featureList.append(featureId)

