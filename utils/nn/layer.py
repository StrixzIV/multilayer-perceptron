import numpy as np
from .neuron import Activation
from .initializer import Initializer

class Layer:

    """Base class for all network layers."""

    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError


    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def parameters(self) -> list:
        return []


    def gradients(self) -> list:
        return []


class Dense(Layer):

    """A fully connected neural network layer."""

    def __init__(self, shape: tuple, activation, initializer: Initializer):
       
        self.input_shape, self.output_shape = shape
        self.weights = initializer.init_weights() if initializer else np.random.randn(*shape) * 0.01
        self.biases = initializer.init_biases((1, self.output_shape)) if initializer else np.zeros((1, self.output_shape))
        self.activation_func = activation
        
        self.input = None
        self.z = None  # Pre-activation output
        
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)


    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:

        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.biases

        if self.activation_func:
            return self.activation_func(self.z)

        return self.z


    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        
        if self.activation_func:

            # Special case for Softmax - gradient is usually passed directly from loss function
            if self.activation_func == Activation.SOFTMAX:
                dz = output_gradient

            else:
    
                activation_derivative = self._get_activation_derivative()

                if activation_derivative:
                    activation_grad = activation_derivative(self.z)
                    dz = output_gradient * activation_grad
                
                # Fallback for unknown activation functions
                else:
                    dz = output_gradient

        else:
            dz = output_gradient

        # Compute gradients
        self.grad_weights = np.dot(self.input.T, dz)
        self.grad_biases = np.sum(dz, axis=0, keepdims=True)
        input_gradient = np.dot(dz, self.weights.T)

        return input_gradient
    

    def _get_activation_derivative(self):
    
        """Get the derivative function for the current activation."""
    
        activation_derivatives = {
            Activation.LINEAR: Activation.LINEAR_derivative,
            Activation.SIGMOID: Activation.SIGMOID_derivative,
            Activation.TANH: Activation.TANH_derivative,
            Activation.RELU: Activation.RELU_derivative,
            Activation.SOFTMAX: Activation.SOFTMAX_derivative
        }
    
        return activation_derivatives.get(self.activation_func)


    def parameters(self) -> list:
        return [self.weights, self.biases]


    def gradients(self) -> list:
        return [self.grad_weights, self.grad_biases]


class Dropout(Layer):


    """
    Dropout layer to prevent overfitting.
    
    During training, it randomly sets a fraction of input units to 0 and
    scales the remaining ones. During evaluation, it does nothing.
    """

    def __init__(self, rate: float, input_size: int):
        self.rate = rate
        self.mask = None


    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:

        if not training:
            return input_data
        
        # Inverted dropout: scale during training, not during testing
        scale_factor = 1.0 / (1.0 - self.rate)
        self.mask = np.random.binomial(1, 1.0 - self.rate, size=input_data.shape) * scale_factor
        return input_data * self.mask


    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.mask
