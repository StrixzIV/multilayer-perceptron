import argparse

import pandas as pd

from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from utils.scaler.ZScoreScaler import ZScoreScaler

from utils.nn.optimizer import Adam
from utils.nn.neuron import Activation
from utils.nn.layer import Dense, Dropout
from utils.nn.losses import binary_crossentropy_loss
from utils.nn.sequential import Sequential, EarlyStopping
from utils.nn.initializer import Initializer, InitializationType

console = Console()
parser = argparse.ArgumentParser(description="A training script for multilayer-perceptron")

parser.add_argument("train_csv", help="Path to training data CSV dataset")
parser.add_argument("vaildate_csv", help="Path to validation data CSV dataset")
parser.add_argument("--label_column_idx", type=int, default=1, help="Index name of label data (target of prediction)")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--output", default="model", help="Output directory to save model and scalers (default: ./model)")

def onehot_encode(target: str) -> list[int, int]:

    '''
        ### Description
        * Benign (B) is represented by one-hot encoding: [1, 0]
        * Malignant (M) is represented by one-hot encoding: [0, 1]
    '''

    return [1, 0] if target == 'B' else [0, 1]


def onehot_decode(encoding: list[int, int]) -> str:

    '''
        ### Description
        * Benign (B) is represented by one-hot encoding: [1, 0]
        * Malignant (M) is represented by one-hot encoding: [0, 1]
    '''
    
    return 'B' if encoding == [1, 0] else 'M'


args = parser.parse_args()

train_csv = args.train_csv
vaildate_csv = args.vaildate_csv
label_column = args.label_column_idx

train_df = pd.read_csv(train_csv, header=None)
vaildate_df = pd.read_csv(vaildate_csv, header=None)

train_df.drop(0, axis=1, inplace=True)
vaildate_df.drop(0, axis=1, inplace=True)

x_train = (train_df.drop(label_column, axis=1)).values
x_validate = (vaildate_df.drop(label_column, axis=1)).values

scaler = ZScoreScaler()
scaler.fit(x_train)

x_train_scaled = (scaler.transform(x_train)).tolist()
x_validate_scaled = (scaler.transform(x_validate)).tolist()

n_classes = 2
n_features = len(x_train[0])

y_train = train_df[1].apply(onehot_encode).tolist()
y_validate = vaildate_df[1].apply(onehot_encode).tolist()

model = Sequential([
    
    Dense(
        shape=(n_features, 64), # Increased from 64
        activation=Activation.RELU,
        initializer=Initializer(fill_type=InitializationType.HE_NORMAL, fan_in=n_features, fan_out=64)
    ),

    Dense(
        shape=(64, 32),
        activation=Activation.RELU,
        initializer=Initializer(fill_type=InitializationType.HE_NORMAL, fan_in=64, fan_out=32)
    ),
   
    Dense(
        shape=(32, 32),
        activation=Activation.RELU,
        initializer=Initializer(fill_type=InitializationType.HE_NORMAL, fan_in=32, fan_out=32)
    ),

    Dense(
        shape=(32, n_classes), # Output layer
        activation=Activation.SIGMOID,
        initializer=Initializer(fill_type=InitializationType.GLOROT_NORMAL, fan_in=32, fan_out=n_classes)
    )
    
])

optimizer = Adam(
    params=model.parameters(), 
    learning_rate=5e-3
)

early_stopping = EarlyStopping(
    patience=100,
    min_delta=1e-3,
    restore_best_weights=True
)

model.summary()

console.print(f"\n[bold green]Starting Training...[/bold green]")
model.fit(
    x_train=x_train_scaled,
    y_train=y_train,
    optimizer=optimizer,
    loss_func=binary_crossentropy_loss,
    epochs=args.epochs,
    batch_size=64,
    metric="accuracy",
    x_validate=x_validate_scaled,
    y_validate=y_validate,
    display_interval=1,
    early_stopping=early_stopping
)

console.print(f"\n[bold blue]Saving model and scaler...[/bold blue]")
model.as_json(f"{args.output}.json")
scaler.to_json(f"{args.output}_scaler.json")

# Final evaluation
console.print(f"\n[bold magenta]Final Evaluation:[/bold magenta]")
final_loss, final_accuracy = model.evaluate(x_validate_scaled, y_validate, 'accuracy')
console.print(f"  Validation Loss: {float(final_loss):.4f}")
console.print(f"  Validation metric: {float(final_accuracy):.4f}")

console.print(f"\n[green]Training completed! Model saved as '{args.output}.json'[/green]")
