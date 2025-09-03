import math
import random

from enum import Enum

from utils.nn.variable import Variable

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

    def __init__(self,
                fill_type: InitializationType = InitializationType.ZEROS,
                stddev: float = 0.2,
                constant_value: float = 0.0,
                fan_in: int = 1,
                fan_out: int = 1
    ):
        self.stddev = stddev
        self.fill_type = fill_type
        self.constant_value = constant_value
        self.fan_in = fan_in
        self.fan_out = fan_out

    
    def __call__(self) -> Variable:

        match self.fill_type:

            case InitializationType.ZEROS:
                return Variable(0.0)
            
            case InitializationType.CONSTANT:
                return Variable(self.constant_value)
            
            case InitializationType.RANDOM_UNIFORM:
                return Variable(random.uniform(a = -0.2, b = 0.2))
            
            case InitializationType.RANDOM_NORMAL:
                return Variable(random.normalvariate(mu = 0, sigma = self.stddev))
            
            case InitializationType.GLOROT_NORMAL:
                stddev = math.sqrt(2.0 / (self.fan_in + self.fan_out))
                return Variable(random.normalvariate(mu=0, sigma=stddev))

            case InitializationType.GLOROT_UNIFORM:
                limit = math.sqrt(6.0 / (self.fan_in + self.fan_out))
                return Variable(random.uniform(a=-limit, b=limit))
            
            case InitializationType.HE_NORMAL:
                stddev = math.sqrt(2.0 / self.fan_in)
                return Variable(random.normalvariate(mu=0, sigma=stddev))

            case InitializationType.HE_UNIFORM:
                limit = math.sqrt(6.0 / self.fan_in)
                return Variable(random.uniform(a=-limit, b=limit))

            case InitializationType.LECUN_NORMAL:
                stddev = math.sqrt(1.0 / self.fan_in)
                return Variable(random.normalvariate(mu=0, sigma=stddev))

            case InitializationType.LECUN_UNIFORM:
                limit = math.sqrt(3.0 / self.fan_in)
                return Variable(random.uniform(a=-limit, b=limit))
            
            case _:
                raise NotImplementedError(f'Weight initialization filler of type {self.fill_type} does not have any implementation yet')