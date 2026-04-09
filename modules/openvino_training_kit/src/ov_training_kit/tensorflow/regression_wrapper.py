import tensorflow as tf
import numpy as np
from .base_wrapper import BaseWrapper
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
)

class RegressionWrapper(BaseWrapper):
    """Wrapper for regression tasks with built-in metrics."""

    def evaluate_mse(self, x, y, batch_size=32):
        """Evaluate Mean Squared Error."""
        y_true, y_pred = self._collect_predictions(x, y, batch_size)
        return mean_squared_error(y_true, y_pred)

    def evaluate_rmse(self, x, y, batch_size=32):
        """Evaluate Root Mean Squared Error."""
        mse = self.evaluate_mse(x, y, batch_size)
        return np.sqrt(mse)

    def evaluate_mae(self, x, y, batch_size=32):
        """Evaluate Mean Absolute Error."""
        y_true, y_pred = self._collect_predictions(x, y, batch_size)
        return mean_absolute_error(y_true, y_pred)

    def evaluate_r2(self, x, y, batch_size=32):
        """Evaluate R² Score."""
        y_true, y_pred = self._collect_predictions(x, y, batch_size)
        return r2_score(y_true, y_pred)

    def evaluate_mape(self, x, y, batch_size=32):
        """Evaluate Mean Absolute Percentage Error."""
        y_true, y_pred = self._collect_predictions(x, y, batch_size)
        return mean_absolute_percentage_error(y_true, y_pred)

    def evaluate_explained_variance(self, x, y, batch_size=32):
        """Evaluate Explained Variance Score."""
        y_true, y_pred = self._collect_predictions(x, y, batch_size)
        return explained_variance_score(y_true, y_pred)

    def evaluate_max_error(self, x, y, batch_size=32):
        """Evaluate Maximum Residual Error."""
        y_true, y_pred = self._collect_predictions(x, y, batch_size)
        return max_error(y_true, y_pred)

    def evaluate_all_metrics(self, x, y, batch_size=32):
        """Evaluate all regression metrics at once."""
        y_true, y_pred = self._collect_predictions(x, y, batch_size)
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred)
        }

    def _collect_predictions(self, x, y, batch_size=32):
        """Helper to collect true and predicted values."""
        preds = self.model.predict(x, batch_size=batch_size)
        y_true = y if isinstance(y, (list, tuple)) else tf.convert_to_tensor(y).numpy().flatten()
        y_pred = preds.flatten() if preds.ndim > 1 else preds
        return np.array(y_true), np.array(y_pred)