from utils.nn.variable import Variable

class Optimizer:

    def __init__(self, params: list[Variable]):
        self.params = params

    
    def reset_gradient(self) -> None:
        for param in self.params:
            param.grad = 0.0

    
    def update(self) -> None:
        raise NotImplementedError('update() cannot be called from base class "Optimizer"')
    

class SGD(Optimizer):

    def __init__(self, params: list[Variable], learning_rate: float = 1e-3):

        super().__init__(params)

        self.step = 0
        self.learning_rate = learning_rate


    def update(self) -> None:

        for param in self.params:
            param.value -= self.learning_rate * param.grad

        self.step += 1