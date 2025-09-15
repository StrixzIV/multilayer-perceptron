import numpy as np

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
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)