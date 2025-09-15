import numpy as np

def binary_crossentropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    
    """
    Calculates the binary cross-entropy loss and its gradient.
    
    Used for binary classification problems with sigmoid output.
    
    Args:
        y_true: The true binary labels (0 or 1).
        y_pred: The predicted probabilities from the model.
        
    Returns:
        A tuple containing the mean loss and the gradient for backpropagation.
    """
    
    # Clip predictions to prevent log(0) which results in NaN
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Calculate the loss averaged over the batch
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Calculate the gradient averaged over the batch
    gradient = (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)
    
    return loss, gradient


def categorical_crossentropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    
    """
    Calculates the categorical cross-entropy loss and its initial gradient.
    
    This function is designed to be used with a Softmax output layer. The
    combined gradient of CCE and Softmax is simply (y_pred - y_true),
    which simplifies backpropagation.
    
    Args:
        y_true: The true one-hot encoded labels.
        y_pred: The predicted probabilities from the model.
        
    Returns:
        A tuple containing the mean loss and the gradient for backpropagation.
    """
    
    # Clip predictions to prevent log(0) which results in NaN
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Calculate the loss averaged over the batch
    loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
    
    # Calculate the gradient averaged over the batch
    gradient = (y_pred - y_true) / len(y_true)
    
    return loss, gradient


def mean_squared_error_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    
    """
    Calculates the mean squared error loss and its gradient.
    
    Commonly used for regression problems where the output is continuous.
    
    Args:
        y_true: The true target values.
        y_pred: The predicted values from the model.
        
    Returns:
        A tuple containing the mean loss and the gradient for backpropagation.
    """
    
    # Calculate the loss averaged over the batch
    loss = np.mean((y_pred - y_true) ** 2)
    
    # Calculate the gradient averaged over the batch
    gradient = 2 * (y_pred - y_true) / len(y_true)
    
    return loss, gradient

def mean_absolute_error_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    
    """
    Calculates the mean absolute error (L1) loss and its gradient.
    
    More robust to outliers than MSE, commonly used for regression.
    
    Args:
        y_true: The true target values.
        y_pred: The predicted values from the model.
        
    Returns:
        A tuple containing the mean loss and the gradient for backpropagation.
    """
    
    # Calculate the loss averaged over the batch
    loss = np.mean(np.abs(y_pred - y_true))
    
    # Calculate the gradient (sign of the difference)
    gradient = np.sign(y_pred - y_true) / len(y_true)
    
    return loss, gradient


LOSS_FUNCTIONS = {
    'mse': mean_squared_error_loss,
    'mean_squared_error': mean_squared_error_loss,
    'mae': mean_absolute_error_loss,
    'mean_absolute_error': mean_absolute_error_loss,
    'binary_crossentropy': binary_crossentropy_loss,
    'categorical_crossentropy': categorical_crossentropy_loss
}

def get_loss_function(name: str):
    """Get loss function by name string."""
    return LOSS_FUNCTIONS.get(name.lower())
