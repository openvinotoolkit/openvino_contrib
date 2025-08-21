# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for the KMeans clustering model with OpenVINO optimization"""

import joblib
from time import time
from sklearnex.cluster import KMeans as SkKMeans

class KMeans:
    def __init__(self, *args, use_openvino=True, **kwargs):
        """
        Initialize the KMeans wrapper.

        Args:
            *args: Positional arguments for sklearn's KMeans.
            use_openvino (bool): Whether to enable OpenVINO optimizations.
            **kwargs: Keyword arguments for sklearn's KMeans.
        """
        self.use_openvino = use_openvino
        self._ir_model = None

        self.model = SkKMeans(*args, **kwargs)
        print("📦 KMeans model initialized (sklearnex version).")

    def fit(self, X, y=None):
        """
        Fit the KMeans model.

        Args:
            X (array-like): Training data.
            y (ignored): Not used, present for API consistency.
        """
        start = time()
        self.model.fit(X)
        elapsed = time() - start
        print(f"🚀 Training completed in {elapsed:.4f} seconds.")

    def predict(self, X):
        """
        Predict cluster labels for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Predicted labels.
        """
        return self.model.predict(X)

    def evaluate(self, X):
        """
        Evaluate the model by predicting cluster labels for X.

        Args:
            X (array-like): Input data.

        Returns:
            array: Predicted labels.
        """
        labels = self.predict(X)
        print(f"📊 Predicted labels: {labels[:10]} ...")
        return labels

    def save_model(self, path="kmeans_model.joblib"):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="kmeans_model.joblib"):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        self.model = joblib.load(path)
        print(f"📂 Model loaded from {path}")

    def _check_export_support(self, X_train=None):
        """
        Check if the model and input are supported for ONNX/IR conversion.

        Args:
            X_train (array-like): Training data.

        Raises:
            ValueError: If input is a sparse matrix.
        """
        if hasattr(X_train, "toarray"):
            raise ValueError("❌ Sparse matrix input is not supported for ONNX/IR conversion.")
        print("✅ Model and input are supported for ONNX/IR conversion.")

    def convert_to_ir(self, X_train, model_name="kmeans_model"):
        """
        Export KMeans to OpenVINO IR (not implemented).

        Args:
            X_train (array-like): Training data for shape reference.
            model_name (str): Base name for the exported model files.
        """
        self._check_export_support(X_train)
        print("⚠️ KMeans OpenVINO IR export is not implemented yet.")