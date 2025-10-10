import unittest
import tensorflow as tf
import sys
import os
import numpy as np

from openvino_kit.tensorflow import ClassificationWrapper

class TestClassificationWrapper(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.wrapper = ClassificationWrapper(self.model)
        self.x = np.random.randn(16, 8).astype(np.float32)
        self.y = np.random.randint(0, 3, (16,))

    def test_fit_and_score(self):
        self.wrapper.fit(self.x, self.y, epochs=1, batch_size=4, verbose=0)
        acc = self.wrapper.score(self.x, self.y, batch_size=4, verbose=0)
        self.assertIsInstance(acc, (float, np.float32, np.float64))

    def test_predict(self):
        preds = self.wrapper.predict(self.x, batch_size=4, verbose=0)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(preds.shape[0], self.x.shape[0])

    def test_evaluate_accuracy(self):
        self.wrapper.fit(self.x, self.y, epochs=1, batch_size=4, verbose=0)
        acc = self.wrapper.evaluate_accuracy(self.x, self.y, batch_size=4)
        self.assertIsInstance(acc, (float, np.float32, np.float64))

if __name__ == "__main__":
    unittest.main()