import tensorflow as tf
from .base_wrapper import BaseWrapper
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
)

class ClassificationWrapper(BaseWrapper):
    """Wrapper for TensorFlow classification tasks with built-in metrics."""

    def evaluate_accuracy(self, x, y, batch_size=32):
        """Evaluate classification accuracy."""
        y_true, y_pred = self._collect_preds(x, y, batch_size)
        return accuracy_score(y_true, y_pred)

    def evaluate_f1(self, x, y, batch_size=32, average="macro"):
        """Evaluate F1 score."""
        y_true, y_pred = self._collect_preds(x, y, batch_size)
        return f1_score(y_true, y_pred, average=average)

    def evaluate_precision(self, x, y, batch_size=32, average="macro"):
        """Evaluate precision score."""
        y_true, y_pred = self._collect_preds(x, y, batch_size)
        return precision_score(y_true, y_pred, average=average)

    def evaluate_recall(self, x, y, batch_size=32, average="macro"):
        """Evaluate recall score."""
        y_true, y_pred = self._collect_preds(x, y, batch_size)
        return recall_score(y_true, y_pred, average=average)

    def evaluate_confusion_matrix(self, x, y, batch_size=32):
        """Return confusion matrix."""
        y_true, y_pred = self._collect_preds(x, y, batch_size)
        return confusion_matrix(y_true, y_pred)

    def evaluate_classification_report(self, x, y, batch_size=32):
        """Return classification report as a string."""
        y_true, y_pred = self._collect_preds(x, y, batch_size)
        return classification_report(y_true, y_pred)

    def evaluate_roc_auc(self, x, y, batch_size=32, average="macro", multi_class="ovr"):
        """Evaluate ROC AUC score (for multi-class, needs probability output)."""
        y_true, y_score = self._collect_probs(x, y, batch_size)
        return roc_auc_score(y_true, y_score, average=average, multi_class=multi_class)

    def evaluate_log_loss(self, x, y, batch_size=32):
        """Evaluate log loss (cross-entropy loss)."""
        y_true, y_score = self._collect_probs(x, y, batch_size)
        return log_loss(y_true, y_score)

    def _collect_preds(self, x, y, batch_size=32):
        """Helper to collect true and predicted labels."""
        preds = self.model.predict(x, batch_size=batch_size)
        if preds.ndim > 1 and preds.shape[1] > 1:
            y_pred = preds.argmax(axis=1)
        else:
            y_pred = (preds > 0.5).astype("int32").flatten()
        y_true = y if isinstance(y, (list, tuple)) else tf.convert_to_tensor(y).numpy().flatten()
        return y_true, y_pred

    def _collect_probs(self, x, y, batch_size=32):
        """Helper to collect true labels and predicted probabilities."""
        probs = self.model.predict(x, batch_size=batch_size)
        y_true = y if isinstance(y, (list, tuple)) else tf.convert_to_tensor(y).numpy().flatten()
        return y_true, probs

    def evaluate_top_k_accuracy(self, x, y, batch_size=32, k=5):
        """Evaluate Top-K accuracy (default k=5)."""
        preds = self.model.predict(x, batch_size=batch_size)
        topk = tf.math.top_k(preds, k=k).indices.numpy()
        y_true = y if isinstance(y, (list, tuple)) else tf.convert_to_tensor(y).numpy().flatten()
        correct = [int(label in topk_row) for label, topk_row in zip(y_true, topk)]
        return sum(correct) / len(correct) if correct else 0.0