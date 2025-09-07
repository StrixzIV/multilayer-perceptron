import json
import random

from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from typing import Callable

from utils.nn import losses, metrics
from utils.nn.variable import Variable
from utils.nn.layer import Dense, Dropout
from utils.nn.optimizer import Optimizer

def is_valid_layer(layer: any):
    return isinstance(layer, (Dense, Dropout))


class Sequential:

    def __init__(self, layers: list[Dense | Dropout]):

        self.layers = layers
        self._validate_layers(layers)

        self.optmizer = None
        self.loss_functiontion = None
        self.metric = None

        self.batch_size = None
        self.epoch = None

        self.x_train = None
        self.y_train = None
        self.x_validate = None
        self.y_validate = None

        self.train_loss_history = []
        self.train_metric_history = []

        self.val_loss_history = []
        self.val_metric_history = []


    def __call__(self, x: list[list[float]]) -> list[list[Variable]]:
        return self.forward_batch(x)


    @staticmethod
    def _validate_layers(layers: list[Dense | Dropout]) -> None:

        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError('Sequential model layer list should not be empty')

        for (idx, layer) in enumerate(layers):
            if not is_valid_layer(layer):
                raise TypeError(f'Invalid type of sequential model layer at idx #{idx} (got {type(layer)})')

        # Layer continuity check
        for idx in range(len(layers) - 1):

            output_size = layers[idx].output_size
            next_input_size = layers[idx + 1].input_size

            if output_size != next_input_size:
                raise ValueError(f'Layer shape mismatched between layer #{idx} and #{idx + 1} (output size not match with input size of next layer)')


    def _validate_data(self,
                       x_train: list[list[float]],
                       y_train: list[list[float]],
                       x_validate: list[list[float]] | None = None,
                       y_validate: list[list[float]] | None = None):

        if not isinstance(x_train, list) or not all(isinstance(x, list) for x in x_train):
            raise ValueError("x_train must be a list of feature lists")

        if not isinstance(y_train, list) or not all(isinstance(y, list) for y in y_train):
            raise ValueError("y_train must be a list of label vectors")

        if len(x_train) != len(y_train):
            raise ValueError(f"Train sample size({len(x_train)}) and Test sample size({len(y_train)}) are mismatch")    
    
        x_len = len(x_train[0])

        for i, x in enumerate(x_train):
            if len(x) != x_len:
                raise ValueError(f"All rows in x_train must have same length; "
                                 f"row 0 is {x_len} but row {i} is {len(x)}")

        y_len = len(y_train[0])
        out_len = self.layers[-1].output_size

        if y_len != out_len:
            raise ValueError(f"Length of label vectors ({y_len}) must match "
                             f"output size of the final layer ({out_len})")
        
        if x_validate or y_validate:

            if not x_validate or not y_validate:
                raise ValueError("Both x_validate and y_validate must be passed together")

            if not isinstance(x_validate, list) or not all(isinstance(x, list) for x in x_validate):
                raise ValueError("x_validate must be a list of feature lists")

            if not isinstance(y_validate, list) or not all(isinstance(y, list) for y in y_validate):
                raise ValueError("y_validate must be a list of label vectors")

            if len(x_validate) != len(y_validate):
                raise ValueError(f"Validate sample x size({len(x_validate)}) and Validate sample y size({len(y_validate)}) are mismatch")

            for i, x in enumerate(x_validate):
                if len(x) != x_len:
                    raise ValueError(f"All rows in x_validate must have same length as X_train; "
                                     f"row 0 is {x_len} but row {i} is {len(x)}")
            if len(y_validate[0]) != y_len:
                raise ValueError("Length of validation label vectors must match training label vectors")


    def _generate_batches(self, x: list[list[float]], y: list[list[float]]):

        n_samples = len(x)
        indices = list(range(n_samples))

        random.shuffle(indices)

        for start in range(0, n_samples, self.batch_size):

            batch_idx = indices[start:start + self.batch_size]

            batch_x = [x[i] for i in batch_idx]
            batch_y = [y[i] for i in batch_idx]

            yield batch_x, batch_y


    def _calculate_metrics(self, y_true: list[list[float]], y_predict: list[list[Variable]]) -> float:

        match self.metric:

            case 'accuracy':
                return metrics.accuracy_score(y_true, y_predict)
            
            case 'f1_score':
                return metrics.f1_score(y_true, y_predict)
            
            case 'r2_score':
                return metrics.r2_score(y_true, y_predict)
            
            case 'mse':
                return losses.mean_squared_error_loss(y_true, y_predict)
            
            case _:
                raise NotImplementedError(f'Metric "{self.metric}" has not been implemented yet')


    def enable_train_mode(self) -> None:
        for layer in self.layers:
            if hasattr(layer, 'enable_train_mode'):
                layer.enable_train_mode()


    def disable_train_mode(self) -> None:
        for layer in self.layers:
            if hasattr(layer, 'disable_train_mode'):
                layer.disable_train_mode()


    def as_json(self, path: str = './model.json') -> None:

        states = [layer.as_dict() for layer in self.layers]

        with open(path, 'w') as f:
            json.dump({ 'layers': states }, f, indent=4)


    def from_json(self, path: str = './model.json') -> None:

        with open(path) as f:
            params = json.load(f)

        layers_states = params["layers"]
        self.layers = [Dense.from_dict(l_state) for l_state in layers_states]


    def forward(self, x: list[float]) -> list[Variable]:

        for layer in self.layers:
            x = layer(x)

        return x
    
    
    def forward_batch(self, x: list[list[float]]) -> list[list[Variable]]:
        y_pred_batch = [self.forward(x_i) for x_i in x]
        return y_pred_batch


    def evaluate(self, x_batch: list[list[float]], y_batch: list[list[float]]) -> tuple[float, float]:

        self.disable_train_mode()

        y_predict = self.forward_batch(x_batch)
        y_flatten = [[var.value for var in var_list] for var_list in y_predict]

        loss = self.loss_function(y_predict, y_batch)
        metric = self._calculate_metrics(y_flatten, y_batch)

        return (loss, metric)
    

    def summary(self) -> None:

        console = Console()

        table = Table(title="Model Architecture Summary", show_header=True, header_style="bold magenta")
        table.add_column("Layer", style="cyan", width=8)
        table.add_column("Type", style="green", width=12)
        table.add_column("Input Shape", style="yellow", width=12)
        table.add_column("Output Shape", style="yellow", width=12)
        table.add_column("Activation", style="blue", width=12)
        table.add_column("Param #", justify="right", style="red", width=10)
        
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_params = len(layer.get_params())
            total_params += layer_params

            if hasattr(layer, 'rate'):
                layer_type = "Dropout"
                activation = f"rate={layer.rate}"
            
            elif hasattr(layer, 'activation'):
                # Dense layer
                layer_type = "Dense"
                activation = layer.activation.value
                if hasattr(layer, 'neurons'):
                    trainable_params += layer_params
            
            else:
                layer_type = "Unknown"
                activation = "N/A"
            
            table.add_row(
                f"{i+1}",
                layer_type,
                f"({layer.input_size}, NULL)",
                f"(NULL, {layer.output_size})",
                activation,
                f"{layer_params:,}"
            )
        
        console.print(table)
        console.print(f"\n[bold]Total params:[/bold] {total_params:,}")
    

    def fit(
        self,
        x_train: list[list[float]],
        y_train: list[list[float]],
        optimizer: Optimizer,
        loss_func: Callable,
        epochs: int,
        batch_size: int = 16,
        metric: str = "accuracy",
        x_validate: list[list[float]] | None = None,
        y_validate: list[list[float]] | None = None,
        display_interval: int = 10,
        max_display_rows: int = 10
    ) -> None:

        self.optimizer = optimizer
        self.loss_function = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.metric = metric

        self._validate_data(x_train, y_train, x_validate, y_validate)

        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        
        console = Console()

        table = Table(title=f"Training Progress (Loss: {loss_func.__name__})")
        table.add_column("Epoch", justify="right", style="cyan")
        table.add_column("Train Loss", justify="right", style="magenta")
        table.add_column(f"Train {self.metric}", justify="right", style="green")

        if x_validate is not None and y_validate is not None:
            table.add_column("Validation Loss", justify="right", style="yellow")
            table.add_column(f"Validation {self.metric}", justify="right", style="red")

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console
        )
        
        self.enable_train_mode()
        task = progress.add_task("[cyan]Training...", total=epochs)
        
        with Live(Group(progress, table), refresh_per_second=10, console=console, vertical_overflow='visible') as live:
            for epoch in range(1, epochs + 1):
                y_true: list[list[float]] = []
                y_pred: list[list[float]] = []
                
                batch_iter = self._generate_batches(x_train, y_train)
                
                for _, (x_batch, y_batch) in enumerate(batch_iter):
                    self.optimizer.reset_gradient()
                    batch_out_values = self.forward_batch(x_batch)

                    batch_loss = self.loss_function(batch_out_values, y_batch)
                    batch_loss.backward()
        
                    y_pred.extend(batch_out_values)
                    y_true.extend(y_batch)
        
                    self.optimizer.update()

                epoch_loss = loss_func(y_true, y_pred)
                train_metric = self._calculate_metrics(y_true, y_pred)

                self.train_loss_history.append(float(epoch_loss))
                self.train_metric_history.append(float(train_metric))
                
                val_loss, val_metric = None, None
                if x_validate is not None and y_validate is not None:
                    val_loss, val_metric = self.evaluate(x_validate, y_validate)
                    self.val_loss_history.append(float(val_loss))
                    self.val_metric_history.append(float(val_metric))

                progress.update(task, advance=1)
                
                if (epoch % display_interval) == 0 or epoch == epochs:

                    row_data = [
                        str(epoch),
                        f"{float(epoch_loss):.4f}",
                        f"{float(train_metric):.4f}"
                    ]

                    if val_loss is not None:
                        row_data.extend([f"{float(val_loss):.4f}", f"{float(val_metric):.4f}"])
                    
                    table.add_row(*row_data)
                    live.refresh()

