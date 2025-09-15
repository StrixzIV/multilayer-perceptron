# utils/nn/sequential.py

import json
import copy
import numpy as np
import matplotlib.pyplot as plt

from . import metrics
from .layer import Dense, Dropout

from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

console = Console()

class EarlyStopping:

    """
    Stops training when a monitored metric has stopped improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = np.inf
        self.best_weights = None
        self.stop_training = False


    def __call__(self, validation_loss: float, model: 'Sequential'):

        """Checks if training should be stopped."""

        if validation_loss < self.best_loss - self.min_delta:

            self.best_loss = validation_loss
            self.counter = 0

            if self.restore_best_weights:
                self.best_weights = [copy.deepcopy(p) for p in model.parameters()]

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop_training = True
                console.print(f"\n[yellow]Early stopping triggered after {self.patience} epochs without improvement.[/yellow]")

                if self.restore_best_weights and self.best_weights is not None:
                    console.print("[yellow]Restoring model weights from the best epoch.[/yellow]")
                    model.set_parameters(self.best_weights)


class Sequential:

    """A sequential model which stacks layers linearly."""

    def __init__(self, layers: list):
        self.layers = layers

    
    def _plot_training_history(self, history: dict, metric_name: str):

        """Plots the training and validation loss and metrics."""
        
        epochs_range = range(1, len(history['loss']) + 1)
        has_validation = 'val_loss' in history
        
        plt.figure(figsize=(12, 5))
        
        # Plot for Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['loss'], '-', label='Training Loss')

        if has_validation:
            plt.plot(epochs_range, history['val_loss'], '-', label='Validation Loss')

        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot for Metric
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history[metric_name], '-', label=f'Training {metric_name.capitalize()}')

        if has_validation:
            plt.plot(epochs_range, history[f'val_{metric_name}'], '-', label=f'Validation {metric_name.capitalize()}')

        plt.title(f'{metric_name.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_history.png')


    def summary(self):

        """Prints a summary of the model's architecture and parameters."""

        table = Table(title="Model Summary")
        table.add_column("Layer (type)", justify="left", style="cyan")
        table.add_column("Output Shape", style="magenta")
        table.add_column("Param #", justify="right", style="green")

        total_params = 0

        for layer in self.layers:
            params = layer.parameters()
            num_params = sum(p.size for p in params)
            total_params += num_params
            output_shape = f"(None, {layer.output_shape})" if hasattr(layer, 'output_shape') else "N/A"
            table.add_row(layer.__class__.__name__, output_shape, str(num_params))

        console.print(table)
        console.print(f"Total params: {total_params:,}")


    def parameters(self) -> list:
        """Gathers all trainable parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]


    def set_parameters(self, params: list):

        """Sets the model's parameters, used for restoring best weights."""

        param_iter = iter(params)

        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = next(param_iter)
                layer.biases = next(param_iter)


    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:

        """Performs a forward pass through all layers."""

        output = x

        for layer in self.layers:
            output = layer.forward(output, training=training)

        return output


    def forward_batch(self, x: np.ndarray, batch_size: int = 32, training: bool = True) -> np.ndarray:

        """
        Performs forward pass in batches for memory efficiency.
        
        Useful for large datasets that don't fit in memory or when you want
        to control memory usage during inference.
        
        Args:
            x: Input data array
            batch_size: Size of each batch for processing
            training: Whether to run in training mode (affects dropout, etc.)
            
        Returns:
            Complete predictions for all input data
        """

        x = np.array(x)
        n_samples = len(x)
        
        # Get output shape by running a small sample
        sample_output = self.forward(x[:1], training=training)
        output_shape = (n_samples,) + sample_output.shape[1:]
        
        # Pre-allocate output array
        all_outputs = np.zeros(output_shape)
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_x = x[i:end_idx]
            batch_output = self.forward(batch_x, training=training)
            all_outputs[i:end_idx] = batch_output
        
        return all_outputs


    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on input data."""
        return self.forward(x, training=False)


    def predict_batch(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:

        """
        Makes predictions in batches for memory efficiency.
        
        Args:
            x: Input data array
            batch_size: Size of each batch for processing
            
        Returns:
            Predictions for all input data
        """

        return self.forward_batch(x, batch_size=batch_size, training=False)


    def backward(self, grad: np.ndarray):

        """Performs a backward pass through all layers in reverse."""

        for layer in reversed(self.layers):
            grad = layer.backward(grad)


    def fit(self, x_train, y_train, optimizer, loss_func, epochs, batch_size, metric,
            x_validate=None, y_validate=None, display_interval=1, early_stopping=None, plot_history: bool = True) -> dict[str, list[float]]:

        """Trains the model on the given dataset."""

        metric_map = {
            "accuracy": metrics.accuracy_score,
            "f1_score": metrics.f1_score,
            "r2_score": metrics.r2_score,
        }

        x_train, y_train = np.array(x_train), np.array(y_train)

        history = {
            'loss': [],
            metric: []
        }

        if x_validate is not None and y_validate is not None:
            history['val_loss'] = []
            history[f'val_{metric}'] = []

        if x_validate is not None:
            x_validate, y_validate = np.array(x_validate), np.array(y_validate)

        n_samples = len(x_train)
        metric_func = metric_map.get(metric.lower())

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(), console=console) as progress:

            epoch_task = progress.add_task("[cyan]Epochs", total=epochs)

            for epoch in range(epochs):

                epoch_loss, epoch_metrics_score = 0, 0

                permutation = np.random.permutation(n_samples)
                x_train_shuffled, y_train_shuffled = x_train[permutation], y_train[permutation]
                
                num_batches = n_samples // batch_size
                batch_task = progress.add_task(f"[magenta]Epoch {epoch+1}/{epochs}", total=num_batches)

                for i in range(0, n_samples, batch_size):
                    x_batch = x_train_shuffled[i : i + batch_size]
                    y_batch = y_train_shuffled[i : i + batch_size]
                    if not len(x_batch): continue

                    y_pred = self.forward(x_batch, training=True)
                    loss, grad = loss_func(y_batch, y_pred)
                    epoch_loss += loss * len(x_batch)

                    self.backward(grad)

                    all_grads = [g for layer in self.layers for g in layer.gradients()]
                    optimizer.update(all_grads)

                    if metric_func:
                        epoch_metrics_score += metric_func(y_batch, y_pred) * len(y_batch)
                    
                    progress.update(batch_task, advance=1)
                
                progress.remove_task(batch_task)
                
                epoch_loss /= n_samples
                epoch_metrics_score /= n_samples
                history['loss'].append(epoch_loss)
                history[metric].append(epoch_metrics_score)
                log_msg = f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - train_{metric}: {epoch_metrics_score:.4f}"

                if x_validate is not None and y_validate is not None:
                    
                    val_loss, val_metric = self.evaluate(x_validate, y_validate, metric_name=metric)
                    
                    history['val_loss'].append(val_loss)
                    history[f'val_{metric}'].append(val_metric)
                    
                    log_msg += f" - val_loss: {val_loss:.4f} - val_{metric}: {val_metric:.4f}"
                    
                    if early_stopping:
                        early_stopping(val_loss, self)
                        if early_stopping.stop_training:
                            progress.update(epoch_task, advance=epochs - epoch)
                            break
                
                if (epoch + 1) % display_interval == 0:
                    console.print(log_msg)
                
                progress.update(epoch_task, advance=1)

        if plot_history:
            self._plot_training_history(history, metric)

        return history


    def evaluate(self, x: np.ndarray, y: np.ndarray, metric_name: str) -> tuple[float, float]:

        """
        Evaluates the model's performance on a given dataset using a specified metric.
        
        Args:
            x: Input data array.
            y: True target values.
            metric_name: The name of the metric to use (e.g., "accuracy", "f1_score").
            
        Returns:
            A tuple containing the mean loss and the computed metric score.
        """
        
        x, y = np.array(x), np.array(y)
        y_pred = self.forward(x, training=False)
        
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y * np.log(y_pred_clipped)) / len(y)
        
        # Dictionary to map metric names to functions
        metric_map = {
            "accuracy": metrics.accuracy_score,
            "f1_score": metrics.f1_score,
            "r2_score": metrics.r2_score,
        }
        
        score = 0.0
        metric_func = metric_map.get(metric_name.lower())
        
        if metric_func:
            score = metric_func(y, y_pred)
        else:
            print(f"Warning: Unknown metric '{metric_name}'. Returning 0.0 for score.")
            
        return loss, score


    def as_json(self, filepath: str):

        """Saves the model architecture and learned weights to a JSON file."""

        model_dict = {"architecture": [], "weights": []}
        for layer in self.layers:

            layer_config = {"class_name": layer.__class__.__name__}

            if isinstance(layer, Dense):

                layer_config["config"] = {
                    "shape": (layer.input_shape, layer.output_shape),
                    "activation": layer.activation_func.__name__ if layer.activation_func else None
                }

                model_dict["weights"].append({
                    "weights": layer.weights.tolist(),
                    "biases": layer.biases.tolist()
                })

            elif isinstance(layer, Dropout):
                layer_config["config"] = {"rate": layer.rate}
            
            model_dict["architecture"].append(layer_config)


        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=4)
