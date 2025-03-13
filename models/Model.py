class Model():
    def __init__(self):
        self.layers = {}
        self.training = True

        self.learning_reate = 1e-3

    def __call__(self, input_ids, *args, **kwargs):
        return self.forward(input_ids, *args, **kwargs)
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, input_ids):
        pass
    
