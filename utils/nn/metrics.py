import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """
    Calculates the accuracy score.
    
    Args:
        y_true: The true labels (one-hot encoded).
        y_pred: The predicted labels (probabilities or raw scores).
        
    Returns:
        The accuracy score as a float between 0 and 1.
    """

    if y_true.ndim > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

    else:
        y_true_labels = y_true
        y_pred_labels = np.round(y_pred)
    
    correct_predictions = np.sum(y_true_labels == y_pred_labels)
    total_samples = len(y_true_labels)
    
    return correct_predictions / total_samples if total_samples > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """
    Calculates the F1 score for binary classification.
    
    Args:
        y_true: The true binary labels (one-hot encoded or single dimension).
        y_pred: The predicted binary labels (probabilities or raw scores).
        
    Returns:
        The F1 score as a float between 0 and 1.
    """

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    else:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # F1 Score
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """
    Calculates the R^2 (R-squared) score for regression.
    
    Args:
        y_true: The true target values.
        y_pred: The predicted values from the model.
        
    Returns:
        The R^2 score.
    """

    mean_y_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y_true) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
    
    return 1 - (ss_residual / ss_total)
