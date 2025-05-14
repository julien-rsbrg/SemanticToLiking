import torch

class WeightClipper():
    def __init__(self,min_value=-1,max_value=1):
        self.min_value = min_value
        self.max_value = max_value
        
    def __call__(self, module):
        if hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data = module.weight.data.clamp(self.min_value,self.max_value)

        for submodule in module.children():
            self(submodule)

class WeightSigmoid():
    def __init__(self,min_value=-1,max_value=1):
        self.min_value = min_value
        self.max_value = max_value
        self.sigmoid_fn = torch.nn.Sigmoid()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data = self.sigmoid_fn(module.weight.data)*(self.max_value - self.min_value) + self.min_value
            
        for submodule in module.children():
            self(submodule)
