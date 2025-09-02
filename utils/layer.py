import random

from enum import Enum
from typing import TypeAlias

from utils.variable import Variable
from utils.neuron import Activation, Neuron
from utils.initializer import InitializationType, Initializer

class LayerType(Enum):
    DENSE = "dense"
    DROPOUT = "dropout"

class Dense:

    def __init__(self,
                 shape: tuple[int, int],
                 activation: Activation,
                 initializer: Initializer = Initializer(fill_type=InitializationType.RANDOM_UNIFORM)):

        self.input_size = shape[0]
        self.output_size = shape[1]

        self.neurons: list[Neuron] = []
        self.activation = activation

        neuron_activation_func = Activation.LINEAR if activation == Activation.SOFTMAX else activation

        for _ in range(self.output_size):
            self.neurons.append(Neuron(
                input_size = self.input_size,
                activation_type = neuron_activation_func,
                initializer = initializer
            ))

    
    def __call__(self, values: list) -> list[Variable]:
        
        output = [neuron(values) for neuron in self.neurons]

        if self.activation == Activation.SOFTMAX:
            output = self._softmax(output)

        return output


    @staticmethod
    def _softmax(output: list[Variable]) -> list[Variable]:
        result = [out.exp() for out in output]
        result_sum = sum(result)
        return [res / result_sum for res in result]
    

    def get_params(self) -> list[Variable]:

        params: list[Variable] = []

        for neuron in self.neurons:
            params.extend(neuron.get_params())
    
        return params


    def as_dict(self) -> dict[str, any]:
        return {
            'layer_type': LayerType.DENSE,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation_type': self.activation.value,
            'parameters': [var.value for var in self.get_params()]
        }
    

    @classmethod
    def from_dict(constructor, state: dict[str, any]) -> 'Dense':

        layer = constructor(
            shape = (state['input_size'], state['output_size']),
            activation = Activation(state['activation_type']),
            initializer = Initializer(fill_type = InitializationType.ZEROS)
        )

        raw_params = state['parameters']
        params = layer.get_params()

        if len(raw_params) != len(params):
            raise ValueError('State parameters size mismatch the current layer parameter size')
        
        for (param, value) in zip(params, raw_params):
            param.value = value

        return layer
    

class Dropout:

    """
    Dropout layer for regularization.

    During training, it randomly sets a fraction `rate` of input units to 0
    at each update and scales the remaining units by 1 / (1 - rate).
    During evaluation, it does nothing and just passes the data through.
    """

    def __init__(self, rate: float):

        if not (0.0 <= rate < 1.0):
            raise ValueError(f"Dropout rate must be in the range [0, 1), but got {rate}")
        
        self.rate = rate
        self.training = True


    def __call__(self, values: list[Variable]) -> list[Variable]:

        # If we are in evaluation mode, do nothing.
        if not self.training:
            return values

        output: list[Variable] = []

        # Inverted dropout: scale the kept neurons during training.
        scale_factor = 1.0 / (1.0 - self.rate)

        for v in values:

            # Drop this neuron (set its output to 0)
            if random.random() < self.rate:
                output.append(v * 0)

            # Keep and scale this neuron
            else:
                output.append(v * scale_factor)
        
        return output


    def get_params(self) -> list[Variable]:
        return []

    def enable_train_mode(self) -> 'Dropout':
        self.training = True
        return self

    def disable_train_mode(self) -> 'Dropout':
        self.training = False
        return self

    def as_dict(self) -> dict[str, any]:
        return {
            'layer_type': LayerType.DROPOUT,
            'rate': self.rate,
        }

    @classmethod
    def from_dict(constructor, state: dict[str, any]) -> 'Dropout':
        return constructor(rate=state['rate'])


Layer: TypeAlias = Dense | Dropout
