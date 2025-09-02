import os
import json
import pickle

import numpy as np
import pandas as pd

class ZScoreScaler:

    def __init__(self, mean: float = np.nan, std: float = np.nan):
        self.std = std
        self.mean = mean


    def fit(self, values: pd.Series) -> None:
        
        self.std = values.std()
        self.mean = values.mean()
        
        if self.std == 0:
            raise ValueError('Standard deviation cannot be zero')


    def transform(self, values: float | pd.Series) -> float | pd.Series:
        
        if np.isnan(self.mean) or np.isnan(self.std):
            raise ValueError("Scaler is not fitted with values yet")
        
        return (values - self.mean) / self.std


    def inverse_transform(self, values: float | pd.Series) -> float | pd.Series:
        
        if np.isnan(self.mean) or np.isnan(self.std):
            raise ValueError("Scaler is not fitted with values yet")
        
        return (values * self.std) + self.mean


    def to_pkl(self, filename: str = 'scaler.pkl') -> None:

        try:

            with open(filename, 'wb') as f:
                pickle.dump(self, f)

            print(f"Scaler successfully saved to {os.path.abspath(filename)}")

        except IOError as e:
            print(f"Error saving scaler to {os.path.abspath(filename)}: {e}")


    def to_json(self, filename: str = 'scaler.json') -> None:

        params = {
            'mean': self.mean,
            'std': self.std
        }

        try:

            with open(filename, 'w') as f:
                json.dump(params, f, indent=4)

            print(f"Parameters successfully saved to {os.path.abspath(filename)}")

        except IOError as e:
            print(f"Error saving parameters to {os.path.abspath(filename)}: {e}")


    def from_json(self, filename: str = 'scaler.json') -> bool:

        try:

            with open(filename, 'r') as f:
                params = json.load(f)
                self.mean = params.get('mean', np.nan)
                self.std = params.get('std', np.nan)

            print(f"Parameters successfully loaded from {filename}")
            return True

        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load parameters from {filename}. Reason: {e}")
            self.mean = np.nan
            self.std = np.nan
            return False


def load_pkl_scaler(filename: str = 'scaler.pkl') -> ZScoreScaler:

    try:

        with open(filename, 'rb') as f:
            scaler = pickle.load(f)

        print(f"Scaler successfully loaded from {os.path.abspath(filename)}")
        return scaler

    except (IOError, pickle.PickleError) as e:
        print(f"Error loading scaler from {os.path.abspath(filename)}: {e}")
        return None
