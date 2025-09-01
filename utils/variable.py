import math

from enum import Enum
from typing import Callable

EPSILON = 1e-10

class AutogradOPS(Enum):
    ADD = "add"
    MULT = "mult"
    POW = "pow"
    EXP = "exp"
    LOG = "log"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"


class Variable:

    def __init__(self, value: int | float, _parents: tuple = (), _op: AutogradOPS = None, label = 'value'):

        self.grad = 0.0
        self.value = value
        self.label = label
        
        self._operation = _op
        self._parents = set(_parents)
        self._backward: Callable[None, None] = lambda: None


    def __repr__(self) -> str:
        return f'Variable({self.value}, label={self.label})'


    def __float__(self) -> float:
        return float(self.value)


    def __neg__(self):
        return self * (-1)


    def __add__(self, other: any):
        
        other_var = other if isinstance(other, Variable) else Variable(other)

        ret_val = self.value + other_var.value
        ret = Variable(ret_val, _parents = (self, other_var), _op = AutogradOPS.ADD)

        def _backward() -> None:
            self.grad += ret.grad
            other_var.grad += ret.grad

        ret._backward = _backward
        return ret
    

    def __sub__(self, other: any):
        return self + (-other)
    

    def __mul__(self, other: any):

        other_var = other if isinstance(other, Variable) else Variable(other)

        ret_val = self.value * other_var.value
        ret = Variable(ret_val, _parents = (self, other_var), _op = AutogradOPS.MULT)

        def _backward() -> None:
            self.grad += other_var.value * ret.grad
            other_var.grad += self.value * ret.grad

        ret._backward = _backward
        return ret
    

    def __truediv__(self, other: any):
        return self * (other ** -1)
    

    def __pow__(self, other: int | float):

        if not isinstance(other, (int, float)):
            raise ValueError('__pow__ operator only supports int or float type')
    
        ret = Variable(self.value ** other, _parents = (self, ), _op = AutogradOPS.POW)

        def _backward() -> None:
            self.grad += (other * self.value ** (other - 1)) * ret.grad

        ret._backward = _backward
        return ret


    def __radd__(self, other: any):
        return self + other
    

    def __rsub__(self, other: any):

        other_var = other

        if not isinstance(other, Variable):
            other_var = Variable(other)

        return other_var + (-self)


    def exp(self):

        result = math.exp(self.value)
        ret = Variable(result, _parents=(self, ), _op = AutogradOPS.EXP)

        def _backward():
            self.grad += result * ret.grad

        ret._backward = _backward
        return ret
    

    def log(self, base: int | float = None):

        result = 0

        if not base:
            result = math.log(self.value + EPSILON)

        else:
            result = math.log(self.value + EPSILON, base=base)
        
        ret = Variable(result, _parents=(self, ), _op=AutogradOPS.LOG)

        def _backward():

            if not base:
                self.grad += (1 / (self.value + EPSILON)) * ret.grad

            else:
                self.grad += (1 / (self.value * math.log(base) + EPSILON)) * ret.grad

        ret._backward = _backward
        return ret
    

    def tanh(self):

        result = (math.exp(2 * self.value) - 1) / (math.exp(2 * self.value) + 1)
        ret = Variable(result, _parents=(self, ), _op=AutogradOPS.TANH)

        def _backward():
            self.grad += (1 - ret.value ** 2) * ret.grad

        ret._backward = _backward
        return ret
    

    def sigmoid(self):

        result = 1 / (1 + math.exp(-self.value))
        ret = Variable(result, _parents=(self, ), _op=AutogradOPS.SIGMOID)

        def _backward():
            self.grad += ((1 - ret.val) * ret.val) * ret.grad

        ret._backward = _backward
        return ret


    def relu(self):

        result = max(0, self.val)
        ret = Variable(result, _parents=(self, ), _op=AutogradOPS.RELU)
        
        def _backward():
            self.grad += int(ret.value > 0) * ret.grad

        ret._backward = _backward
        return ret
    

    def backward(self) -> None:

        self.grad = 1

        topo: list[Variable] = []
        visited = set()

        def generate_topo(variable: Variable) -> None:

            if variable in visited:
                return
            
            visited.add(variable)

            for parent in variable._parents:
                generate_topo(parent)

            topo.append(variable)

        generate_topo(self)

        for node in reversed(topo):
            node._backward()
