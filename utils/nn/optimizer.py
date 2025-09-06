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
            adaptive_lr = self.learning_rate / (math.sqrt(self.squared_grad_avg[i]) + self.epsilon)
            param.value -= adaptive_lr * param.grad
        
        self.step += 1


class NesterovMomentum(Optimizer):

    def __init__(self, 
                 params: list[Variable], 
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9):

        super().__init__(params)

        self.step = 0
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.squared_grad_avgelocity = [0.0 for _ in params]


    def update(self) -> None:
        
        for i, param in enumerate(self.params):
            
            # Update velocity: v = momentum * v + learning_rate * grad
            self.squared_grad_avgelocity[i] = (
                self.momentum * self.squared_grad_avgelocity[i] + 
                self.learning_rate * param.grad
            )
            
            # Nesterov update: param = param - momentum * v - learning_rate * grad
            param.value -= (
                self.momentum * self.squared_grad_avgelocity[i] + 
                self.learning_rate * param.grad
            )
        
        self.step += 1


class Adam(Optimizer):

    def __init__(self, 
                 params: list[Variable], 
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = EPSILON):
        
        super().__init__(params)

        self.step = 0
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # RMSProp + NestrovMomentum
        self.momentum = [0.0 for _ in params]
        self.squared_grad_avg = [0.0 for _ in params]


    def update(self) -> None:

        self.step += 1
        
        for i, param in enumerate(self.params):

            # Update momentum and squared gradient
            self.momentum[i] = self.beta1 * self.momentum[i] + (1 - self.beta1) * param.grad
            self.squared_grad_avg[i] = self.beta2 * self.squared_grad_avg[i] + (1 - self.beta2) * param.grad ** 2
            
            # Bias correction on momentum and squared gradient
            m_corrected = self.momentum[i] / (1 - self.beta1 ** self.step)
            v_corrected = self.squared_grad_avg[i] / (1 - self.beta2 ** self.step)

            param.value -= self.learning_rate * m_corrected / (math.sqrt(v_corrected) + self.epsilon)
