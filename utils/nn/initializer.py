import numpy as np

from enum import Enum

class InitializationType(Enum):
    ZEROS = "zeros"
    CONSTANT = "constant"
    RANDOM_NORMAL = "normal"
    RANDOM_UNIFORM = "uniform"
    GLOROT_NORMAL = "glorot_normal"
    GLOROT_UNIFORM = "glorot_uniform"
    HE_NORMAL = "he_normal" 
    HE_UNIFORM = "he_uniform" 
    LECUN_NORMAL = "lecun_normal"
    LECUN_UNIFORM = "lecun_uniform"

class Initializer:

    """
    Initializes weights and biases for a Dense layer.
    
    Attributes:
        fill_type (InitializationType): The type of initialization to use.
        fan_in (int): The number of input units to the layer.
        fan_out (int): The number of output units from the layer.
    """

    def __init__(self, fill_type: InitializationType, fan_in: int = None, fan_out: int = None, **kwargs):
        self.fill_type = fill_type
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.constant_value = kwargs.get('constant_value', 0.1)


    def init_weights(self) -> np.ndarray:

        """Initialize weights based on the specified initialization type."""

        shape = (self.fan_in, self.fan_out)

        # --- Basic Initializers ---
        if self.fill_type == InitializationType.ZEROS:
            return np.zeros(shape)

        elif self.fill_type == InitializationType.CONSTANT:
            return np.full(shape, self.constant_value)

        elif self.fill_type == InitializationType.RANDOM_NORMAL:
            return np.random.randn(*shape) * 0.01

        elif self.fill_type == InitializationType.RANDOM_UNIFORM:
            return np.random.uniform(-0.05, 0.05, shape)

        if self.fan_in is None or self.fan_out is None:
             raise ValueError(f"Initialization type '{self.fill_type.value}' requires 'fan_in' and 'fan_out'.")

        # Glorot (Xavier) Initializers - good for tanh, sigmoid, softmax
        if self.fill_type == InitializationType.GLOROT_UNIFORM:
            limit = np.sqrt(6 / (self.fan_in + self.fan_out))
            return np.random.uniform(-limit, limit, shape)
        
        elif self.fill_type == InitializationType.GLOROT_NORMAL:
            stddev = np.sqrt(2 / (self.fan_in + self.fan_out))
            return np.random.normal(0, stddev, shape)

        # He Initializers - the recommended choice for ReLU activations
        elif self.fill_type == InitializationType.HE_NORMAL:
            stddev = np.sqrt(2 / self.fan_in)
            return np.random.normal(0, stddev, shape)
        
        elif self.fill_type == InitializationType.HE_UNIFORM:
            limit = np.sqrt(6 / self.fan_in)
            return np.random.uniform(-limit, limit, shape)

        # LeCun Initializers - a precursor to Glorot
        elif self.fill_type == InitializationType.LECUN_NORMAL:
            stddev = np.sqrt(1 / self.fan_in)
            return np.random.normal(0, stddev, shape)

        elif self.fill_type == InitializationType.LECUN_UNIFORM:
            limit = np.sqrt(3 / self.fan_in)
            return np.random.uniform(-limit, limit, shape)

        else:
            raise ValueError(f"Unknown initialization type: {self.fill_type}")


    def init_biases(self, shape: tuple) -> np.ndarray:
        """Initializes biases as a zero vector."""
        return np.zeros(shape)
