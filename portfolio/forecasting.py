"""Causal lagged-return features and rolling per-asset LASSO forecasts."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig
from .splits import validate_dates


_DEFAULT_ALPHAS = np.logspace(-6, -2, 21)


@dataclass(frozen=True)
class ForecastData:
    """Complete target rows and their per-asset causal feature arrays."""

    features: np.ndarray
    targets: pd.DataFrame


def _validate_returns(returns):
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")
    validate_dates(returns)
    if returns.columns.has_duplicates or returns.shape[1] < 1:
        raise ValueError("returns need at least one uniquely named asset")
    try:
        values = returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be numeric") from exc
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("returns must contain a complete finite date-asset grid")
    if len(returns) < 64:
        raise ValueError("at least 64 return rows are required")
    return values


def build_supervised(returns):
    """Build six features known before each target date.

    The first 63 return rows are feature warm-up. Any missing value anywhere
    is rejected, so later gaps cannot be mistaken for additional warm-up.
    """
    values = _validate_returns(returns)
    n, p = values.shape
    features = np.empty((n - 63, p, 6), dtype=float)
    equal_return = values.mean(axis=1)
    for output_row, target_position in enumerate(range(63, n)):
        history = values[:target_position]
        features[output_row, :, 0] = history[-1]
        features[output_row, :, 1] = np.prod(1.0 + history[-5:], axis=0) - 1.0
        features[output_row, :, 2] = np.prod(1.0 + history[-21:], axis=0) - 1.0
        features[output_row, :, 3] = np.prod(1.0 + history[-63:], axis=0) - 1.0
        features[output_row, :, 4] = np.std(history[-21:], axis=0, ddof=1)
        features[output_row, :, 5] = np.prod(1.0 + equal_return[target_position - 21:target_position]) - 1.0
    if not np.isfinite(features).all():
        raise ValueError("feature construction produced a non-finite value")
    return ForecastData(features, returns.iloc[63:].copy())


def _validate_data(data):
    if not isinstance(data, ForecastData) or not isinstance(data.targets, pd.DataFrame):
        raise ValueError("data must be ForecastData")
    validate_dates(data.targets)
    features = np.asarray(data.features, dtype=float)
    expected = (len(data.targets), data.targets.shape[1], 6)
    if features.shape != expected or data.targets.columns.has_duplicates:
        raise ValueError(f"features must have shape {expected}")
    if not np.isfinite(features).all() or not np.isfinite(data.targets.to_numpy(dtype=float)).all():
        raise ValueError("ForecastData must be finite and complete")


def _validate_config(config):
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    if config.half_life is not None:
        raise ValueError("forecast training requires equal observation weights")


def _training_positions(index, prediction_position, config):
    window = config.window
    if window.observations is not None:
        if prediction_position < window.observations:
            raise ValueError(
                f"Need {window.observations} complete targets before {index[prediction_position].date()}"
            )
        return np.arange(prediction_position - window.observations, prediction_position)
    cutoff = index[prediction_position] - pd.DateOffset(years=window.years)
    left = index.searchsorted(cutoff, side="left")
    positions = np.arange(left, prediction_position)
    if len(positions) < window.min_observations:
        raise ValueError(
            f"Need {window.min_observations} complete targets before {index[prediction_position].date()}"
        )
    return positions


def _fit_predict(features, targets, prediction_features, alpha):
    p = targets.shape[1]
    predictions = np.empty(p, dtype=float)
    constant_counts = {}
    for asset, column in enumerate(targets.columns):
        X = features[:, asset, :]
        scaler = StandardScaler().fit(X)
        constant_counts[column] = int(np.count_nonzero(scaler.var_ == 0.0))
        model = Lasso(alpha=alpha, max_iter=100000, tol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(scaler.transform(X), targets.iloc[:, asset].to_numpy())
        predictions[asset] = model.predict(
            scaler.transform(np.asarray(prediction_features[asset])[None, :])
        )[0]
    return predictions, constant_counts


def _validate_alpha(alpha):
    if isinstance(alpha, bool) or not np.isscalar(alpha):
        raise ValueError("alpha must be a finite positive number")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive number")
    return alpha


def forecast_at(data, date, config, alpha):
    """Refit each asset through the prior target row and forecast ``date``."""
    _validate_data(data)
    _validate_config(config)
    alpha = _validate_alpha(alpha)
    date = pd.Timestamp(date)
    try:
        position = data.targets.index.get_loc(date)
    except KeyError as exc:
        raise ValueError("date must be an available complete target date") from exc
    if not isinstance(position, (int, np.integer)):
        raise ValueError("date must identify exactly one target row")
    train_positions = _training_positions(data.targets.index, int(position), config)
    predictions, constant_counts = _fit_predict(
        data.features[train_positions],
        data.targets.iloc[train_positions],
        data.features[position],
        alpha,
    )
    diagnostics = {
        "date": date,
        "alpha": alpha,
        "train_start": data.targets.index[train_positions[0]],
        "train_end": data.targets.index[train_positions[-1]],
        "train_count": len(train_positions),
        "constant_feature_counts": constant_counts,
    }
    return pd.Series(predictions, index=data.targets.columns, name=date), diagnostics


def calibrate_alpha(data, config, *, end, alphas=None, n_splits=3, test_size=21):
    """Choose one shared alpha from fixed late chronological validation folds."""
    _validate_data(data)
    _validate_config(config)
    if (isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1
            or isinstance(test_size, bool) or not isinstance(test_size, int) or test_size < 1):
        raise ValueError("n_splits and test_size must be positive integers")
    candidates = _DEFAULT_ALPHAS if alphas is None else np.asarray(alphas, dtype=float)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("alphas must be a nonempty one-dimensional collection")
    candidates = np.asarray([_validate_alpha(value) for value in candidates], dtype=float)
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("alphas must be unique")

    end = pd.Timestamp(end)
    eligible_count = int(data.targets.index.searchsorted(end, side="right"))
    if eligible_count == 0 or end < data.targets.index[0] or end > data.targets.index[-1]:
        raise ValueError("end must fall within the complete target date range")
    validation_count = n_splits * test_size
    first_validation = eligible_count - validation_count
    if first_validation <= 0:
        raise ValueError("not enough complete targets for the requested folds")

    rows = []
    for fold in range(n_splits):
        validation_start = first_validation + fold * test_size
        validation_positions = np.arange(validation_start, validation_start + test_size)
        train_positions = _training_positions(data.targets.index, validation_start, config)
        for alpha in candidates:
            predictions = np.empty((test_size, data.targets.shape[1]), dtype=float)
            for asset in range(data.targets.shape[1]):
                X = data.features[train_positions, asset, :]
                scaler = StandardScaler().fit(X)
                model = Lasso(alpha=float(alpha), max_iter=100000, tol=1e-8)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(scaler.transform(X), data.targets.iloc[train_positions, asset])
                predictions[:, asset] = model.predict(
                    scaler.transform(data.features[validation_positions, asset, :])
                )
            residuals = predictions - data.targets.iloc[validation_positions].to_numpy()
            rows.append({
                "alpha": float(alpha),
                "fold": fold + 1,
                "mse": float(np.mean(residuals ** 2)),
                "train_start": data.targets.index[train_positions[0]],
                "train_end": data.targets.index[train_positions[-1]],
                "train_count": len(train_positions),
                "validation_start": data.targets.index[validation_positions[0]],
                "validation_end": data.targets.index[validation_positions[-1]],
                "validation_count": len(validation_positions),
            })
    scores = pd.DataFrame(rows)
    means = scores.groupby("alpha", sort=False)["mse"].mean()
    minimum = float(means.min())
    tied = means.index[np.isclose(means.to_numpy(), minimum, atol=1e-12, rtol=1e-8)]
    return float(np.max(tied)), scores
