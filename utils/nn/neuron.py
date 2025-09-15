import numpy as np

class Activation:
    """Container for activation functions and their derivatives."""
    
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