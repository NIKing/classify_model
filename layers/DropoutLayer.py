import numpy as np

class DropoutLayer():
    def __init__(self, rate=1.0):
        self.p = rate

    def __call__(self, _input: np.ndarray):

        # 以概率随机子网络
        self.d = np.random.binomial(n=1, p=self.p, size=_input.shape).astype(np.float32)
        
        # 使用逆暂退法处理输出数据
        return _input * self.d / self.p
