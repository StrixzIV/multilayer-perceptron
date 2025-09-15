# utils/nn/sequential.py

import numpy as np
import json
import copy

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

    def backward(self, grad: np.ndarray):
        """Performs a backward pass through all layers in reverse."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, x_train, y_train, optimizer, loss_func, epochs, batch_size, metric,
            x_validate=None, y_validate=None, display_interval=1, early_stopping=None):
        """Trains the model on the given dataset."""
        x_train, y_train = np.array(x_train), np.array(y_train)
        if x_validate is not None:
            x_validate, y_validate = np.array(x_validate), np.array(y_validate)

        n_samples = len(x_train)

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(), console=console) as progress:
            epoch_task = progress.add_task("[cyan]Epochs", total=epochs)

            for epoch in range(epochs):
                epoch_loss, epoch_accuracy = 0, 0
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

                    if metric == "accuracy":
                        epoch_accuracy += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                    
                    progress.update(batch_task, advance=1)
                
                progress.remove_task(batch_task)
                
                epoch_loss /= n_samples
                epoch_accuracy /= n_samples
                log_msg = f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}"

                if x_validate is not None and y_validate is not None:
                    val_loss, val_accuracy = self.evaluate(x_validate, y_validate)
                    log_msg += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                    
                    if early_stopping:
                        early_stopping(val_loss, self)
                        if early_stopping.stop_training:
                            progress.update(epoch_task, advance=epochs - epoch)
                            break
                
                if (epoch + 1) % display_interval == 0:
                    console.print(log_msg)
                
                progress.update(epoch_task, advance=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Evaluates the model's performance on a given dataset."""
        x, y = np.array(x), np.array(y)
        y_pred = self.forward(x, training=False)
        
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y * np.log(y_pred_clipped)) / len(y)
        
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

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