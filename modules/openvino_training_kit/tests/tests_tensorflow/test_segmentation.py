import unittest
import tensorflow as tf
import sys
import os
import numpy as np

from openvino_kit.ensorflow import SegmentationWrapper

class TestSegmentationWrapper(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3), padding='same'),
            tf.keras.layers.Conv2D(2, 1, activation='softmax', padding='same')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.wrapper = SegmentationWrapper(self.model)
        self.x = np.random.randn(8, 32, 32, 3).astype(np.float32)
        self.y = np.random.randint(0, 2, (8, 32, 32))

    def test_fit_and_score(self):
        self.wrapper.fit(self.x, self.y, epochs=1, batch_size=2, verbose=0)
        score = self.wrapper.score(self.x, self.y, batch_size=2, verbose=0)
        self.assertIsInstance(score, (float, np.float32, np.float64))

    def test_predict(self):
        preds = self.wrapper.predict(self.x, batch_size=2, verbose=0)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(preds.shape[0], self.x.shape[0])

    def test_evaluate_iou(self):
        self.wrapper.fit(self.x, self.y, epochs=1, batch_size=2, verbose=0)
        iou = self.wrapper.evaluate_iou(self.x, self.y, num_classes=2, batch_size=2)
        self.assertIsInstance(iou, (float, np.float32, np.float64))

if __name__ == "__main__":
    unittest.main()