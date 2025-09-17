import numpy as np
import pandas as pd

from rich.console import Console

from utils.nn.sequential import Sequential
from utils.scaler.ZScoreScaler import ZScoreScaler
from utils.nn.losses import binary_crossentropy_loss

console = Console()

def onehot_encode(target: str) -> list[int, int]:

    '''
        ### Description
        * Benign (B) is represented by one-hot encoding: [1, 0]
        * Malignant (M) is represented by one-hot encoding: [0, 1]
    '''

    return [1, 0] if target == 'B' else [0, 1]


test_df = pd.read_csv('./validation.csv', header=None)
test_df.drop(0, axis = 1, inplace = True)

x_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].apply(onehot_encode).to_list()

scaler = ZScoreScaler()
scaler.from_json('./model_scaler.json')

x_test_scaled = (scaler.transform(x_test)).tolist()

model = Sequential.from_json("model.json")
model.summary()

y_pred_probs = model.predict(x_test_scaled)

y_true_np = np.array(y_test)
y_pred_np = np.array(y_pred_probs)

loss, _ = binary_crossentropy_loss(y_true_np, y_pred_np)

console.print(f"Binary Crossentropy Loss: {loss:.4f}")
