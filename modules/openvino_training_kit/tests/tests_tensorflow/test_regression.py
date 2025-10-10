import unittest
import tensorflow as tf
import sys
import os
import numpy as np

from openvino_kit.tensorflow import RegressionWrapper

class TestRegressionWrapper(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.wrapper = RegressionWrapper(self.model)
        self.x = np.random.randn(20, 5).astype(np.float32)
        self.y = np.random.randn(20, 1).astype(np.float32)

    def test_fit_and_score(self):
        self.wrapper.fit(self.x, self.y, epochs=1, batch_size=4, verbose=0)
        score = self.wrapper.score(self.x, self.y, batch_size=4, verbose=0)
        self.assertIsInstance(score, (float, np.float32, np.float64))

    def test_predict(self):
        preds = self.wrapper.predict(self.x, batch_size=4, verbose=0)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(preds.shape[0], self.x.shape[0])

    def test_evaluate_mse(self):
        self.wrapper.fit(self.x, self.y, epochs=1, batch_size=4, verbose=0)
        mse = self.wrapper.evaluate_mse(self.x, self.y, batch_size=4)
        self.assertIsInstance(mse, (float, np.float32, np.float64))

if __name__ == "__main__":
    unittest.main()