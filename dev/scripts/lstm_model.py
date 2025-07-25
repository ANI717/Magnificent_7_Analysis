import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out.squeeze()


def train_lstm_model(model, X_train, y_train, X_val, y_val, device="cpu", epochs=20, batch_size=32, lr=1e-3):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("hidden_size", model.hidden_size)
        mlflow.log_param("num_layers", model.num_layers)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item() * xb.size(0)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Log final metrics
        mlflow.log_metric("final_train_loss", train_loss)
        mlflow.log_metric("final_val_loss", val_loss)

        # Log model
        mlflow.pytorch.log_model(model, "lstm_model")

    return model
