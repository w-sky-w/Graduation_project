from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


@dataclass
class FingerflexDataset:
    data: np.ndarray  # (time, channels)
    flex: np.ndarray  # (time, 5)
    cue: np.ndarray   # (time,)
    subject_id: str



def load_fingerflex_mat(mat_path: str | Path) -> FingerflexDataset:
    mat_path = Path(mat_path)
    mat = loadmat(mat_path)
    data = np.asarray(mat["data"], dtype=np.float32)
    flex = np.asarray(mat["flex"], dtype=np.float32)
    cue = np.asarray(mat["cue"]).squeeze().astype(np.int32)

    if data.ndim != 2:
        raise ValueError(f"Expected data to be 2D (time, channels), got {data.shape}")
    if flex.ndim != 2 or flex.shape[1] != 5:
        raise ValueError(f"Expected flex to have shape (time, 5), got {flex.shape}")
    if data.shape[0] != flex.shape[0] or data.shape[0] != cue.shape[0]:
        raise ValueError("time dimension mismatch among data/flex/cue")

    subject_id = mat_path.stem.split("_")[0]
    return FingerflexDataset(data=data, flex=flex, cue=cue, subject_id=subject_id)



def make_time_lagged_features(
    data: np.ndarray,
    target: np.ndarray,
    lag_ms: int = 300,
    step_ms: int = 50,
    sampling_rate: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create lagged features by concatenating channel values from past windows.

    Example: lag=300ms, step=50ms gives 6 lag points per channel.
    """
    lag_samples = int(lag_ms * sampling_rate / 1000)
    step_samples = int(step_ms * sampling_rate / 1000)
    if lag_samples <= 0 or step_samples <= 0:
        raise ValueError("lag_ms and step_ms must be positive")

    offsets = np.arange(0, lag_samples, step_samples)
    t_start = offsets.max()
    num_rows = data.shape[0] - t_start
    x = np.empty((num_rows, data.shape[1] * len(offsets)), dtype=np.float32)

    for i, t in enumerate(range(t_start, data.shape[0])):
        lagged = [data[t - off] for off in offsets]
        x[i] = np.concatenate(lagged, axis=0)

    y = target[t_start:]
    return x, y



def split_by_time(x: np.ndarray, y: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(x)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return (x[:i1], y[:i1]), (x[i1:i2], y[i1:i2]), (x[i2:], y[i2:])



def apply_channel_dropout(
    x: np.ndarray,
    channels: int,
    lag_points: int,
    drop_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if drop_ratio <= 0:
        return x.copy(), np.array([], dtype=int)

    drop_count = max(1, int(round(channels * drop_ratio)))
    drop_idx = rng.choice(channels, size=drop_count, replace=False)

    x_masked = x.copy()
    for ch in drop_idx:
        for lag in range(lag_points):
            feat_idx = lag * channels + ch
            x_masked[:, feat_idx] = 0.0
    return x_masked, np.sort(drop_idx)



def corrcoef_per_finger(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = np.zeros(y_true.shape[1], dtype=np.float32)
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) < 1e-8 or np.std(y_pred[:, i]) < 1e-8:
            out[i] = np.nan
        else:
            out[i] = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
    return out



def run_baseline(
    mat_path: str | Path,
    dropout_ratio: float = 0.0,
    seed: int = 42,
    alpha: float = 1.0,
    lag_ms: int = 300,
    step_ms: int = 50,
) -> dict:
    ds = load_fingerflex_mat(mat_path)
    x, y = make_time_lagged_features(ds.data, ds.flex, lag_ms=lag_ms, step_ms=step_ms)

    lag_points = int(np.ceil(lag_ms / step_ms))
    channels = ds.data.shape[1]

    (x_tr, y_tr), (_, _), (x_te, y_te) = split_by_time(x, y)

    rng = np.random.default_rng(seed)
    x_tr, dropped = apply_channel_dropout(x_tr, channels, lag_points, dropout_ratio, rng)
    x_te, _ = apply_channel_dropout(x_te, channels, lag_points, dropout_ratio, rng=np.random.default_rng(seed))

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)

    model = MultiOutputRegressor(Ridge(alpha=alpha, random_state=seed))
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)

    r2_each = r2_score(y_te, y_pred, multioutput="raw_values")
    corr_each = corrcoef_per_finger(y_te, y_pred)

    return {
        "subject_id": ds.subject_id,
        "dropout_ratio": dropout_ratio,
        "dropped_channels": dropped.tolist(),
        "r2_mean": float(np.nanmean(r2_each)),
        "corr_mean": float(np.nanmean(corr_each)),
        **{f"r2_{name}": float(v) for name, v in zip(FINGER_NAMES, r2_each)},
        **{f"corr_{name}": float(v) for name, v in zip(FINGER_NAMES, corr_each)},
    }



def parse_dropouts(values: Iterable[str]) -> list[float]:
    return [float(v) for v in values]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline finger flex prediction")
    parser.add_argument("--mat", type=str, required=True, help="Path to ##_fingerflex.mat")
    parser.add_argument("--dropout", type=float, default=0.0, help="Electrode dropout ratio [0,1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lag-ms", type=int, default=300)
    parser.add_argument("--step-ms", type=int, default=50)
    args = parser.parse_args()

    metrics = run_baseline(
        mat_path=args.mat,
        dropout_ratio=args.dropout,
        seed=args.seed,
        alpha=args.alpha,
        lag_ms=args.lag_ms,
        step_ms=args.step_ms,
    )
    for k, v in metrics.items():
        print(f"{k}: {v}")
