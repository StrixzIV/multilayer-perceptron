import math

from utils.nn.variable import Variable, EPSILON

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


class RMSProp(Optimizer):

    def __init__(self, 
                 params: list[Variable], 
                 learning_rate: float = 1e-3,
                 alpha: float = 0.99,
                 epsilon: float = EPSILON):

        super().__init__(params)

        self.step = 0
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon

        self.squared_grad_avg = [0.0 for _ in params]


    def update(self) -> None:
        
        for i, param in enumerate(self.params):

            # Average squared graident
            self.squared_grad_avg[i] = (
                self.alpha * self.squared_grad_avg[i] + 
                (1 - self.alpha) * param.grad ** 2
            )

            # Adapt learning rate based on the average squared graident
            adaptive_lr = self.learning_rate / (math.sqrt(self.squared_grad_avg[i]) + self.eps)
            param.value -= adaptive_lr * param.grad
        
        self.step += 1
