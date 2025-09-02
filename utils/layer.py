from dataclasses import dataclass

from utils.variable import Variable
from utils.neuron import Activation, Neuron
from utils.initializer import InitializationType, Initializer

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

    
    def __call__(self, values: list):
        
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
