from enum import Enum

from utils.nn.variable import Variable
from utils.nn.initializer import Initializer

class Activation(Enum):
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    SOFTMAX = "softmax"

class Neuron:

    def __init__(self, input_size: int, activation_type: Activation, initializer: Initializer):

        self.bias = initializer()
        self.weights = [initializer() for _ in range(input_size)]

        self.activation_type = activation_type
        self.params = self.weights + [self.bias]


    def __call__(self, x: list[Variable | float]) -> Variable:

        # y = \sigma(wx) + b
        weighted_sum = Variable(sum([(w_i * x_i) for (w_i, x_i) in zip(self.weights, x)]))
        weighted_sum += self.bias

        match self.activation_type:

            case Activation.LINEAR:
                return weighted_sum
            
            case Activation.TANH:
                return weighted_sum.tanh()
            
            case Activation.SIGMOID:
                return weighted_sum.sigmoid()
            
            case Activation.RELU:
                return weighted_sum.relu()
            
            case _:
                raise NotImplementedError(f'Activation of type {self.activation_type} does not have any implementation yet')


    def get_params(self) -> list[Variable]:
        return self.params