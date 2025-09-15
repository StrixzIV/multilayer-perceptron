import numpy as np

class Activation:

    """Container for activation functions and their derivatives."""

    @staticmethod
    def LINEAR(x: np.ndarray) -> np.ndarray:
        """Linear activation function (identity function)."""
        return x
    
    @staticmethod
    def LINEAR_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of the linear function."""
        return np.ones_like(x)

    @staticmethod
    def SIGMOID(x: np.ndarray) -> np.ndarray:

        """Sigmoid activation function."""

        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))


    @staticmethod
    def SIGMOID_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid function."""
        sigmoid_x = Activation.SIGMOID(x)
        return sigmoid_x * (1 - sigmoid_x)


    @staticmethod
    def TANH(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)


    @staticmethod
    def TANH_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of the tanh function."""
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2


    @staticmethod
    def RELU(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit activation function."""
        return np.maximum(0, x)


    @staticmethod
    def RELU_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of the RELU function."""
        return np.where(x > 0, 1, 0)


    @staticmethod
    def SOFTMAX(x: np.ndarray) -> np.ndarray:
        """
        Softmax activation function for multi-class classification.
        Includes a stability trick by subtracting the max value.
        """
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    

    @staticmethod
    def SOFTMAX_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of Softmax function.
        Note: This is typically not used directly as Softmax is usually
        combined with cross-entropy loss for more stable computation.
        """
        softmax_x = Activation.SOFTMAX(x)
        return softmax_x * (1 - softmax_x)


    @staticmethod
    def get_activation(name: str):

        """Get activation function by name string."""

        activation_map = {
            'linear': Activation.LINEAR,
            'sigmoid': Activation.SIGMOID,
            'tanh': Activation.TANH,
            'relu': Activation.RELU,
            'softmax': Activation.SOFTMAX
        }

        return activation_map.get(name.lower())


    @staticmethod
    def get_derivative(name: str):
    
        """Get activation derivative function by name string."""
    
        derivative_map = {
            'linear': Activation.LINEAR_derivative,
            'sigmoid': Activation.SIGMOID_derivative,
            'tanh': Activation.TANH_derivative,
            'relu': Activation.RELU_derivative,
            'softmax': Activation.SOFTMAX_derivative
        }
    
        return derivative_map.get(name.lower())
