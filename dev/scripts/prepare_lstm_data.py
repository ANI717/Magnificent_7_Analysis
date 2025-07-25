import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import torch


def prepare_lstm_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "return",
    sequence_length: int = 10,
    test_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:

    # Sort by date to maintain time order
    df = df.sort_index()

    # Keep only selected columns
    df = df[feature_cols + [target_col]].dropna()

    # Normalize features and target
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Build sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length, :-1])   # All features except target
        y.append(data_scaled[i+sequence_length, -1])       # Target comes after the window

    X, y = np.array(X), np.array(y)

    # Split into train/test
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, scaler
