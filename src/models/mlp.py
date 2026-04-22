"""MLP on flattened sequence. Configurable hidden layers, dropout, seed."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPModel:
    """
    MLP: (seq_len * n_features) -> 1. Optional BatchNorm and dropout.
    Fit accepts optional validation set for early stopping; otherwise splits from train.

    Training never materialises the full dataset as a single PyTorch tensor.
    Each mini-batch is loaded from the numpy (or memmap) arrays on demand, so
    peak RSS during training is O(batch_size), not O(n_samples).
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
        for h in self.hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Internal helpers: load a contiguous batch from a numpy/memmap array.
    # For memmap X the OS pages in only the requested rows; for a regular
    # in-RAM array the copy is a few hundred KB — negligible either way.
    # ------------------------------------------------------------------

    @staticmethod
    def _load_batch_x(X: np.ndarray, idx: np.ndarray) -> torch.Tensor:
        """Fancy-index X at idx and return a float32 CPU tensor (flat)."""
        n = len(idx)
        # ascontiguousarray forces a real copy (not a view) so torch can own it.
        batch = np.ascontiguousarray(X[idx].reshape(n, -1), dtype=np.float32)
        return torch.from_numpy(batch)

    @staticmethod
    def _load_slice_x(X: np.ndarray, s: int, e: int) -> torch.Tensor:
        """Contiguous slice [s:e] of X → flat float32 tensor."""
        batch = np.ascontiguousarray(X[s:e].reshape(e - s, -1), dtype=np.float32)
        return torch.from_numpy(batch)

    @staticmethod
    def _load_batch_y(y: np.ndarray, idx: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(y[idx], dtype=np.float32)).unsqueeze(1)

    @staticmethod
    def _load_slice_y(y: np.ndarray, s: int, e: int) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(y[s:e], dtype=np.float32)).unsqueeze(1)

    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """
        X, y: (n, seq_len, n_features), (n,). May be numpy.memmap.
        If X_val/y_val provided, use for early stopping; otherwise reserve last 20%.
        """
        if X_val is None or y_val is None:
            n = len(X)
            val_size = max(1024, n // 5)
            # Use the chronologically last slice as validation (preserves time order).
            X_val = X[n - val_size:]
            y_val = y[n - val_size:]
            X = X[: n - val_size]
            y = y[: n - val_size]

        n_train = len(X)

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
            perm = self._rng.permutation(n_train)
            epoch_loss = 0.0
            for start in range(0, n_train, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                bx_t = self._load_batch_x(X, batch_idx).to(self.device)
                by_t = self._load_batch_y(y, batch_idx).to(self.device)
                pred = self._net(bx_t)
                loss = criterion(pred, by_t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item() * len(batch_idx)
            epoch_loss /= n_train

            self._net.eval()
            with torch.no_grad():
                val_loss = self._batched_mse(X_val, y_val, criterion)
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
        preds: list[np.ndarray] = []
        eval_bs = max(self.batch_size, 8192)
        with torch.no_grad():
            for s in range(0, len(X), eval_bs):
                e = min(s + eval_bs, len(X))
                bx_t = self._load_slice_x(X, s, e).to(self.device)
                out = self._net(bx_t).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0).squeeze(1).astype(np.float32)

    def _batched_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        criterion: nn.Module,
    ) -> float:
        """Compute validation MSE without loading the full set into a single tensor."""
        if self._net is None:
            raise RuntimeError("Model not fitted")
        n = len(X)
        eval_bs = max(self.batch_size, 8192)
        total = 0.0
        for s in range(0, n, eval_bs):
            e = min(s + eval_bs, n)
            bx_t = self._load_slice_x(X, s, e).to(self.device)
            by_t = self._load_slice_y(y, s, e).to(self.device)
            total += criterion(self._net(bx_t), by_t).item() * (e - s)
        return total / n

    def get_state_dict(self) -> dict:
        if self._net is None:
            raise RuntimeError("Model not fitted")
        return {k: v.cpu().clone() for k, v in self._net.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        if self._net is None:
            self._net = self._build_net().to(self.device)
        self._net.load_state_dict(state)
        self._net.to(self.device)
