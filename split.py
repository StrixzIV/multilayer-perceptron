#!/usr/bin/env python3

import sys
import argparse

import pandas as pd

from rich.console import Console
from sklearn.model_selection import train_test_split

console = Console()
parser = argparse.ArgumentParser(description="A training script for ft_linear_regression")

parser.add_argument("csv_path", help="Path to CSV dataset")
parser.add_argument("--train-out-name", default="train.csv", help="Output filename for output splitted train data(default: ./train.csv)")
parser.add_argument("--validate-out-name", default="validation.csv", help="Output filename for output splitted validation data(default: ./validation.csv)")
parser.add_argument("--val-size", type=float, default=0.2, help="Number of training epochs")

args = parser.parse_args()

csv_path = args.csv_path
val_size = args.val_size
train_out_name = args.train_out_name
validate_out_name = args.validate_out_name

try:
    df = pd.read_csv(csv_path)
    
except Exception as e:
    console.print(f"[bold red]Error:[/] Failed to open '{args.csv_path}'. Reason: {e}")
    sys.exit(1)

(train, validate) = train_test_split(
    df, 
    test_size = val_size, 
    shuffle = True
)

train.to_csv(train_out_name, index = False)
validate.to_csv(validate_out_name, index = False)
