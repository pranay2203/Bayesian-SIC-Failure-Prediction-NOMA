"""NumPy MC-dropout neural classifier; stochastic inference approximates Bayesian uncertainty."""
import random
import numpy as np


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class MCDropoutNet:
    def __init__(self, n_features, seed=42, hidden_units=32, dropout_rate=0.25):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / n_features), (n_features, hidden_units)); self.b1 = np.zeros(hidden_units)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_units), hidden_units); self.b2 = 0.0
        self.dropout_rate = dropout_rate

    def forward(self, x, training=False, rng=None):
        pre = x @ self.w1 + self.b1; hidden = np.maximum(0, pre); mask = None
        if training:
            mask = (rng.random(hidden.shape) >= self.dropout_rate).astype(float) / (1 - self.dropout_rate)
            hidden *= mask
        return hidden @ self.w2 + self.b2, hidden, pre, mask


def train_model(x_train, y_train, epochs, batch_size, lr, seed):
    """Train a one-hidden-layer classifier with mini-batch gradient descent."""
    set_seed(seed); model = MCDropoutNet(x_train.shape[1], seed); rng = np.random.default_rng(seed); losses = []
    for _ in range(epochs):
        order = rng.permutation(len(x_train)); epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]; x, y = x_train[ids], y_train[ids]
            logits, hidden, pre, mask = model.forward(x, training=True, rng=rng); prob = _sigmoid(logits)
            epoch_loss += -np.sum(y * np.log(prob + 1e-8) + (1 - y) * np.log(1 - prob + 1e-8))
            dlogits = (prob - y) / len(y); grad_w2 = hidden.T @ dlogits; grad_b2 = dlogits.sum()
            dpre = (dlogits[:, None] * model.w2[None, :] * mask) * (pre > 0)
            model.w1 -= lr * (x.T @ dpre); model.b1 -= lr * dpre.sum(axis=0)
            model.w2 -= lr * grad_w2; model.b2 -= lr * grad_b2
        losses.append(epoch_loss / len(x_train))
    return model, losses


def predict_with_uncertainty(model, x, mc_samples=40):
    """Repeated dropout predictions yield a mean probability and uncertainty (std)."""
    rng = np.random.default_rng(2026); x = np.asarray(x, dtype=np.float64)
    draws = np.vstack([_sigmoid(model.forward(x, training=True, rng=rng)[0]) for _ in range(mc_samples)])
    return draws.mean(axis=0), draws.std(axis=0), draws
