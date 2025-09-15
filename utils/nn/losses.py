import numpy as np

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