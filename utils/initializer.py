import random

from enum import Enum

from utils.variable import Variable

class InitializationType(Enum):
    ZEROS = "zeros"
    UNIFORM = "uniform"
    NORMAL = "normal"


class Initializer:

    def __init__(self, fill_type: InitializationType = InitializationType.ZEROS, stddev: float = 0.2):
        self.stddev = stddev
        self.fill_type = fill_type

    
    def __call__(self) -> float:

        match self.fill_type:

            case InitializationType.ZEROS:
                return Variable(0.0)
            
            case InitializationType.NORMAL:
                return Variable(random.normalvariate(mu = 0, sigma = self.stddev))
            
            case InitializationType.UNIFORM:
                return Variable(random.uniform(a = -0.2, b = 0.2))
            
            case _:
                raise NotImplementedError(f'Weight initialization filler of type {self.fill_type} does not have any implementation yet')