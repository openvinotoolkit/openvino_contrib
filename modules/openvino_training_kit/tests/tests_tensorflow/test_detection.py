import unittest
import tensorflow as tf
import sys
import os
import numpy as np

from openvino_kit.tensorflow import DetectionWrapper

class TestDetectionWrapper(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(4)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.wrapper = DetectionWrapper(self.model)
        self.x = np.random.randn(12, 10).astype(np.float32)
        self.y = np.random.randn(12, 4).astype(np.float32)

    def test_fit_and_score(self):
        self.wrapper.fit(self.x, self.y, epochs=1, batch_size=3, verbose=0)
        # Use MSE as metric for detection/regression tasks
        def mse_metric(preds, targets):
            return tf.keras.losses.mse(targets, preds).numpy().mean()
        score = self.wrapper.score(self.x, self.y, metric_fn=mse_metric, batch_size=3, verbose=0)
        self.assertIsInstance(score, (float, np.float32, np.float64))

    def test_predict(self):
        preds = self.wrapper.predict(self.x, batch_size=3, verbose=0)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(preds.shape[0], self.x.shape[0])

if __name__ == "__main__":
    unittest.main()