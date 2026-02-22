"""MLP on flattened sequence. Configurable hidden layers, dropout, seed."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPModel:
    """
    MLP: (seq_len * n_features) -> 1. Optional BatchNorm and dropout.
    Fit accepts optional validation set for early stopping; otherwise splits from train.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden: tuple[int, ...] = (128, 64, 32),
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
        self.input_dim = seq_len * n_features
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self._net: nn.Module | None = None
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._rng = np.random.default_rng(seed)

    def _build_net(self) -> nn.Module:
        layers = []
        d = self.input_dim
        for i, h in enumerate(self.hidden):
            layers.append(nn.Linear(d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """
        X, y: (n, seq_len, n_features), (n,). If X_val/y_val provided, use for early stopping.
        Otherwise reserve 20% of X,y as validation.
        """
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
        X_t = torch.from_numpy(X_flat).to(self.device)
        y_t = torch.from_numpy(y).float().unsqueeze(1).to(self.device)

        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1).astype(np.float32)
            X_val_t = torch.from_numpy(X_val_flat).to(self.device)
            y_val_t = torch.from_numpy(y_val).float().unsqueeze(1).to(self.device)
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
        self._net = self._build_net().to(self.device)
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
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.from_numpy(X_flat).to(self.device))
        return out.cpu().numpy().squeeze(1).astype(np.float32)

    def get_state_dict(self) -> dict:
        if self._net is None:
            raise RuntimeError("Model not fitted")
        return {k: v.cpu().clone() for k, v in self._net.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        if self._net is None:
            self._net = self._build_net().to(self.device)
        self._net.load_state_dict(state)
        self._net.to(self.device)
