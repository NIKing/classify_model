import numpy as np

class Adam():
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.learning_rate = lr


