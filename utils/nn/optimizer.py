import numpy as np

class SGD:

    """
    Stochastic Gradient Descent optimizer.
    
    The most basic optimizer that updates parameters in the direction
    opposite to the gradient, scaled by the learning rate.
    """

    def __init__(self, params: list, learning_rate: float = 0.01):
        self.params = params
        self.learning_rate = learning_rate


    def update(self, grads: list):

        """
        Updates the model parameters using the calculated gradients.
        
        Args:
            grads: A list of gradient arrays corresponding to the model parameters.
        """

        for param, grad in zip(self.params, grads):
            param -= self.learning_rate * grad


class SGDMomentum:

    """
    SGD with Momentum optimizer.
    
    Accumulates a velocity vector in directions of persistent reduction
    in the objective across iterations.
    """

    def __init__(self, params: list, learning_rate: float = 0.01, momentum: float = 0.9):

        self.params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize velocity vectors
        self.velocity = [np.zeros_like(p) for p in self.params]


    def update(self, grads: list):
        """
        Updates the model parameters using momentum-based gradients.
        
        Args:
            grads: A list of gradient arrays corresponding to the model parameters.
        """
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update velocity: v = momentum * v - learning_rate * grad
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            param += self.velocity[i]


class NesterovMomentum:

    """
    Nesterov Accelerated Gradient (NAG) optimizer.
    
    A variant of momentum that performs the gradient calculation at the
    "looked-ahead" position, often providing better convergence.
    """

    def __init__(self, params: list, learning_rate: float = 0.01, momentum: float = 0.9):

        self.params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize velocity vectors
        self.velocity = [np.zeros_like(p) for p in self.params]

    def update(self, grads: list):

        """
        Updates the model parameters using Nesterov momentum.
        
        Args:
            grads: A list of gradient arrays corresponding to the model parameters.
        """

        for i, (param, grad) in enumerate(zip(self.params, grads)):

            # Store previous velocity
            v_prev = self.velocity[i].copy()
            
            # Update velocity
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            param += -self.momentum * v_prev + (1 + self.momentum) * self.velocity[i]


class RMSprop:

    """
    RMSprop (Root Mean Square Propagation) optimizer.
    
    Adapts the learning rate for each parameter by dividing by a running
    average of the magnitudes of recent gradients.
    """

    def __init__(self, params: list, learning_rate: float = 0.001, decay: float = 0.9, epsilon: float = 1e-8):

        self.params = params
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize squared gradient accumulation
        self.square_avg = [np.zeros_like(p) for p in self.params]

    def update(self, grads: list):

        """
        Updates the model parameters using RMSprop adaptive learning rates.
        
        Args:
            grads: A list of gradient arrays corresponding to the model parameters.
        """

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.square_avg[i] = self.decay * self.square_avg[i] + (1 - self.decay) * (grad ** 2)
            param -= self.learning_rate * grad / (np.sqrt(self.square_avg[i]) + self.epsilon)


class Adam:

    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines the ideas of Momentum and RMSprop to efficiently update
    network weights.
    """

    def __init__(self, params: list, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):

        self.params = params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        # Initialize moment vectors
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]


    def update(self, grads: list):

        """
        Updates the model parameters using the calculated gradients.
        
        Args:
            grads: A list of gradient arrays corresponding to the model parameters.
        """

        self.t += 1

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update biased first moment estimate (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate (RMSprop)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
