class Model():
    def __init__(self):
        self.layers = {}
        self.training = True

    def __call__(self, input_ids, *args, **kwargs):
        return self.forward(input_ids, *args, **kwargs)
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, input_ids):
        pass
    
