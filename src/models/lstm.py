"""LSTM for next-price prediction. Configurable hidden size, layers, dropout, seed."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _LSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        last = self.ln(out[:, -1])
        return self.fc(last).squeeze(-1)


class LSTMModel:
    """
    LSTM over (seq_len, n_features) -> scalar. Fit accepts optional X_val/y_val for early stopping.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        device: str = "cpu",
        max_epochs: int = 50,
        patience: int = 7,
        batch_size: int = 256,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self._net: _LSTMNet | None = None
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y).float().to(self.device)

        if X_val is not None and y_val is not None:
            X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
            y_val_t = torch.from_numpy(y_val).float().to(self.device)
        else:
            n = X_t.size(0)
            val_size = max(1024, n // 5)
            perm = self._rng.permutation(n)
            X_t = X_t[perm]
            y_t = y_t[perm]
            X_val_t = X_t[-val_size:]
            y_val_t = y_t[-val_size:]
            X_t = X_t[:-val_size]
            y_t = y_t[:-val_size]

        torch.manual_seed(self.seed)
        self._net = _LSTMNet(
            self.n_features,
            hidden=self.hidden,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3
        )
        criterion = nn.MSELoss()

        self._patience_counter = 0
        self._best_loss = float("inf")
        best_state = None

        for epoch in range(self.max_epochs):
            self._net.train()
            perm = self._rng.permutation(X_t.size(0))
            X_tr = X_t[perm]
            y_tr = y_t[perm]
            epoch_loss = 0.0
            for start in range(0, X_tr.size(0), self.batch_size):
                end = min(start + self.batch_size, X_tr.size(0))
                pred = self._net(X_tr[start:end])
                loss = criterion(pred, y_tr[start:end])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item() * (end - start)
            epoch_loss /= X_tr.size(0)

            self._net.eval()
            with torch.no_grad():
                val_pred = self._net(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            scheduler.step(val_loss)
            if val_loss < self._best_loss:
                self._best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                self._patience_counter = 0
            else:
                self._patience_counter += 1
            if self._patience_counter >= self.patience:
                break
        if best_state is not None:
            self._net.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not fitted")
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.from_numpy(X.astype(np.float32)).to(self.device))
        return out.cpu().numpy().astype(np.float32)

    def get_state_dict(self) -> dict:
        if self._net is None:
            raise RuntimeError("Model not fitted")
        return {k: v.cpu().clone() for k, v in self._net.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        if self._net is None:
            self._net = _LSTMNet(
                self.n_features,
                hidden=self.hidden,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
        self._net.load_state_dict(state)
        self._net.to(self.device)
