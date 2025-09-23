# multilayer-perceptron

A complete implementation of a multi-layer perceptron neural network built from scratch using NumPy for binary classification of breast cancer data (Benign vs Malignant).

## Project Structure

```
.
├── Makefile
├── README.md
├── data
│   └── data.csv
├── mnist_computer_vision.ipynb
├── predict.py
├── requirements.txt
├── split.py
├── train.py
├── universal_approximator.ipynb
├── utils
│   ├── __init__.py
│   ├── nn
│   │   ├── initializer.py
│   │   ├── layer.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── neuron.py
│   │   ├── optimizer.py
│   │   └── sequential.py
│   └── scaler
│       ├── ZScoreScaler.py
│       └── __init__.py
└── validation.csv
```

## Quick Start

### Training a Model
```bash
python train.py train.csv validation.csv
```

### Making Predictions
```bash
python predict.py data/data_test.csv
```

---

## 1. Feedforward Process

### Theory
The feedforward process is how information flows through the network from input to output. Each layer transforms the input using a linear transformation followed by a non-linear activation function.

### Mathematical Foundation

The feedforward process computes the output of a neural network by propagating input data through successive layers.

For a single neuron:
$$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$
$$a = f(z)$$

In matrix form for a layer:
$$Z = XW + B$$
$$A = f(Z)$$

Where:

  - $X \in \mathbb{R}^{m \times n}$ (m samples, n features)
  - $W \in \mathbb{R}^{n \times h}$ (n inputs, h hidden units)
  - $B \in \mathbb{R}^{1 \times h}$ (bias vector)
  - $Z \in \mathbb{R}^{m \times h}$ (pre-activation)
  - $A \in \mathbb{R}^{m \times h}$ (post-activation)

Multi-layer composition:
$$A^{(0)} = X \text{ (input)}$$
$$Z^{(1)} = A^{(0)}W^{(1)} + B^{(1)}$$
$$A^{(1)} = f^{(1)}(Z^{(1)})$$
$$\dots$$
$$Z^{(L)} = A^{(L-1)}W^{(L)} + B^{(L)}$$
$$A^{(L)} = f^{(L)}(Z^{(L)}) = \hat{y} \text{ (output)}$$

#### Universal Approximation Theorem
The Universal Approximation Theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $\mathbb{R}^n$ to arbitrary accuracy.

*Proof sketch:* For any continuous function g on [0,1] and $\epsilon \> 0$, there exists a neural network with sigmoid activation that approximates g within $\epsilon$. This follows from the density of linear combinations of sigmoid functions in the space of continuous functions.

-----

## 2\. Gradient Descent

### Theory
Gradient descent is the optimization algorithm that minimizes the loss function by iteratively updating parameters in the direction opposite to the gradient.

---

### Mathematical Foundation

Gradient descent minimizes a loss function $L(\theta)$ by iteratively updating parameters in the direction of the negative gradient.

#### Basic Update Rule
$$\theta_{t+1} = \theta_t - \alpha\cdot\nabla L(\theta_t)$$

Where:

  - $\theta$ = parameters (weights and biases)
  - $\alpha$ = learning rate
  - $\nabla L(\theta)$ = gradient of loss with respect to parameters

#### Convergence Proof for Convex Functions

For a convex function L with L-Lipschitz gradients and learning rate $\alpha \le 1/L$:

$$
L(\theta_{t+1}) \le L(\theta_t) - \alpha\nabla L(\theta_t)^T\nabla L(\theta_t) + (\alpha L/2)||\nabla L(\theta_t)||^2
$$
$$L(\theta_{t+1}) = L(\theta_t) - \alpha(1 - \alpha L/2)||\nabla L(\theta_t)||^2$$

#### Stochastic Gradient Descent (SGD)

Instead of computing the full gradient, SGD uses mini-batch approximations:

$$
\nabla L(\theta) \approx \left(\frac{1}{|B|}\right) \sum_{i \in B} \nabla L(\theta; x_i, y_i)
$$

#### Adam (Adaptive Moment Estimation)
Adam optmizer is a combination of Nestrov's momentum and RMSProp to create adaptive learning rates

**First moment (Nestrov's momentum)**
$$
m_t = \beta_1m_{t-1} + g_t(1 - \beta_1)
$$

**Second moment (RMSProp)**
$$
v_t = \beta_2v_{t-1} + g_t^2(1 - \beta_2)
$$

**Bias correction**

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t = \frac{v_t}{1-\beta_2^t}
$$

**Parameter update**

$$
\theta_{t+1} = \theta_t - \alpha\left(\frac{\hat m_t}{\sqrt{\hat v_t + \epsilon}}\right)
$$

-----

### Implementation Details

**1. SGD**
```python
def update(self, grads: list):
    for param, grad in zip(self.params, grads):
        param -= self.learning_rate * grad
```

**2. Adam Optimizer**
```python
def update(self, grads: list):

    self.t += 1

    for i, (param, grad) in enumerate(zip(self.params, grads)):
        # Momentum term
        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
        # RMSprop term
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m[i] / (1 - self.beta1 ** self.t)
        v_hat = self.v[i] / (1 - self.beta2 ** self.t)
        
        # Parameter update
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

```

## 3\. Backpropagation

### Theory
Backpropagation computes gradients by applying the chain rule of calculus, propagating error signals backward through the network.

### Mathematical Foundation

For a composition of functions $f(g(x))$, the derivative is:
$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

For Neural Networks:
Let L be the loss function, and consider layer l with:

  - Input: $a^{(l-1)}$
  - Weights: $W^{(l)}$
  - Biases: $b^{(l)}$
  - Pre-activation: $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$
  - Post-activation: $a^{(l)} = f(z^{(l)})$

Error Propagation:
Define error terms: $\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$

For output layer L:
$$\delta^{(L)} = \nabla_a L \odot f'(z^{(L)})$$

For hidden layers ($l = L-1, L-2, \dots, 1$):
$$\delta^{(l)} = ((W^{(l+1)})^T\delta^{(l+1)}) \odot f'(z^{(l)})$$

Gradient Computation:
$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^T$$
$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

Mathematical Proof of Backpropagation:
Using the chain rule:
$$\frac{\partial L}{\partial W^{(l)}_{ij}} = \frac{\partial L}{\partial z^{(l)}_j} \cdot \frac{\partial z^{(l)}_j}{\partial W^{(l)}_{ij}}$$
$$= \delta^{(l)}_j \cdot a^{(l-1)}_i$$

For the error propagation:
$$\frac{\partial L}{\partial z^{(l)}_i} = \sum_j \left(\frac{\partial L}{\partial z^{(l+1)}_j} \cdot \frac{\partial z^{(l+1)}_j}{\partial z^{(l)}_i}\right)$$
$$= \sum_j (\delta^{(l+1)}_j \cdot W^{(l+1)}_{ji} \cdot f'(z^{(l)}_i))$$
$$= f'(z^{(l)}_i) \sum_j W^{(l+1)}_{ji}\delta^{(l+1)}_j$$

Computational Complexity:

  - Forward pass: $O(W)$ where W is total number of weights
  - Backward pass: $O(W)$ - same complexity as forward pass
  - Total: $O(W)$ per training example

Special Case - Softmax + Cross-Entropy:
For softmax output with cross-entropy loss:
$$\delta^{(L)} = \hat{y} - y$$
This elegant result occurs because:
$$\frac{\partial}{\partial z_i}[-\sum_j y_j\log(\text{softmax}(z)_j)] = \text{softmax}(z)_i - y_i = \hat{y}_i - y_i$$

**Implementation of Backpropagation (`layer.py`):**
```python
def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    if self.activation_func:
        if self.activation_func == Activation.SOFTMAX:
            dz = output_gradient  # Special case for softmax + cross-entropy
        else:
            activation_derivative = self._get_activation_derivative()
            if activation_derivative:
                activation_grad = activation_derivative(self.z)
                dz = output_gradient * activation_grad  # Chain rule
            else:
                dz = output_gradient
    else:
        dz = output_gradient

    # Compute gradients
    self.grad_weights = np.dot(self.input.T, dz)
    self.grad_biases = np.sum(dz, axis=0, keepdims=True)
    input_gradient = np.dot(dz, self.weights.T)
    
    return input_gradient
```

**Activation Derivatives (`neuron.py`):**
```python
@staticmethod
def RELU_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

@staticmethod
def SIGMOID_derivative(x: np.ndarray) -> np.ndarray:
    sigmoid_x = Activation.SIGMOID(x)
    return sigmoid_x * (1 - sigmoid_x)
```

**Loss Function Gradients (`losses.py`):**
```python
def binary_crossentropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    # Clip to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    gradient = (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)
    
    return loss, gradient
```

**Network-wide Backpropagation (`sequential.py`):**
```python
def backward(self, grad: np.ndarray):
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
```

**Example: Training Loop (`sequential.py`):**
```python
y_pred = self.forward(x_batch, training=True)
loss, grad = loss_func(y_batch, y_pred)

self.backward(grad)
all_grads = [g for layer in self.layers for g in layer.gradients()]
optimizer.update(all_grads)
```

-----

## 4\. Overfitting and how to prevent it

### Mathematical Foundation

Overfitting occurs when a model learns the training data too well, capturing noise rather than the underlying pattern. This results in high training accuracy but poor generalization.

#### Bias-Variance Decomposition
The expected prediction error can be decomposed as:
$$E[(y - \hat{y})^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Where:

  - **Bias²**: Error from oversimplifying assumptions
  - **Variance**: Error from sensitivity to small fluctuations in training data
  - **Irreducible Error**: Noise inherent in the data

Overfitting occurs when **Variance \>\> Bias**

### Regularization Techniques

#### 1.  Early Stopping Mathematical Justification:
Let $L_{\text{train}}(\theta, t)$ and $L_{\text{validate}}(\theta, t)$ be training and validation losses at epoch t.

**Theorem:**

If $L_{\text{train}}$ decreases monotonically while $L_{\text{val}}$ increases after epoch $t^*$, then the model is overfitting for **t \>\> \*t**.

```python
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

    def __call__(self, validation_loss: float, model: 'Sequential'):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = [copy.deepcopy(p) for p in model.parameters()]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
```

#### Adding Dropout Layer

Dropout randomly sets neurons to zero with probability p during training.
Expected Value Preservation:

$$E[\text{dropout}(x)] = (1-p) \cdot x$$

During inference, we scale by (1-p) to maintain expected activations:

$${\hat y}(x) = (1-p) \cdot x$$

```python
class Dropout(Layer):

    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
   
        if not training:
            return input_data
        
        # Inverted dropout: scale during training
        scale_factor = 1.0 / (1.0 - self.rate)
        self.mask = np.random.binomial(1, 1.0 - self.rate, size=input_data.shape) * scale_factor
   
        return input_data * self.mask
```

#### Change Weight Initialization method

Xavier/Glorot Initialization:

For layers with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs:
$$\text{Var}(W) = \frac{2}{(n_{\text{in}} + n_{\text{out}})}$$

Derivation: To maintain activation variance across layers:

- Forward: $\text{Var}(z) = n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)$
- Backward: $\text{Var}(\nabla x) = n_{\text{out}} \cdot \text{Var}(W) \cdot \text{Var}(\nabla z)$

He Initialization (for ReLU):
$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$
Derivation: ReLU kills negative activations, so only \~50% of neurons are active:
$$\text{Var}(\text{ReLU}(z)) \approx \frac{1}{2} \cdot \text{Var}(z)$$
