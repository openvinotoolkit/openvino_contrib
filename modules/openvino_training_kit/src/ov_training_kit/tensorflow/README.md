# OpenVINO Kit - TensorFlow Integration

Wrappers for TensorFlow/Keras models with OpenVINO for inference, quantization, and deployment.

## Available Wrappers

- **BaseWrapper**: Core functionality for all TensorFlow models
- **ClassificationWrapper**: Built-in metrics for image/text classification
- **RegressionWrapper**: Specialized for regression tasks (MSE, MAE, R²)
- **SegmentationWrapper**: Semantic segmentation with IoU, Dice metrics
- **DetectionWrapper**: Object detection with custom metric support

## Quick Start

```python
import tensorflow as tf
from ov_training_kit.tensorflow import ClassificationWrapper

# 1. Create and train your model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 2. Wrap your model
wrapper = ClassificationWrapper(model)

# 3. Train normally
wrapper.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# 4. Convert to optimized OpenVINO format
example_input = tf.random.normal((1, 224, 224, 3))
wrapper.convert_to_ov(example_input=example_input)

# 5. Setup OpenVINO engine and compile for fast inference
wrapper.setup_core(cache_dir="./cache")
wrapper.compile(device="CPU")

# 6. Run fast inference with OpenVINO
result = wrapper.infer({0: example_input})
predicted_class = tf.argmax(result[list(result.keys())[0]], axis=1)
print(f"Predicted class: {predicted_class}")
```

**What happened?** Your TensorFlow model is now running optimized OpenVINO inference - typically 2-5x faster on Intel CPUs!

## Classification Metrics (Built-in)

```python
classifier = ClassificationWrapper(model)

# Train first
classifier.fit(x_train, y_train, epochs=5)

# Get comprehensive metrics with one line each
accuracy = classifier.evaluate_accuracy(x_test, y_test)
f1_score = classifier.evaluate_f1(x_test, y_test, average='macro')
precision = classifier.evaluate_precision(x_test, y_test, average='macro')
recall = classifier.evaluate_recall(x_test, y_test, average='macro')

# Advanced metrics
confusion_matrix = classifier.evaluate_confusion_matrix(x_test, y_test)
roc_auc = classifier.evaluate_roc_auc(x_test, y_test, multi_class='ovr')
top5_accuracy = classifier.evaluate_top_k_accuracy(x_test, y_test, k=5)

print(f"Accuracy: {accuracy:.3f} | F1: {f1_score:.3f} | Top-5: {top5_accuracy:.3f}")
```

## Quantization (Smaller, Faster Models)

**Option 1: Post-Training Quantization (PTQ)**
```python
# Train first, then compress
wrapper.fit(x_train, y_train, epochs=5)
wrapper.quantize()  # Model is now ~4x smaller
wrapper.convert_to_ov(example_input=example_input)

# Check if accuracy is still good
accuracy_after = wrapper.evaluate_accuracy(x_test, y_test)
print(f"Quantized model accuracy: {accuracy_after:.3f}")
```

**Option 2: Quantization-Aware Training (QAT)**
```python
# Quantize first, then fine-tune
wrapper.quantize()  # Prepare for QAT
wrapper.fit(x_train, y_train, epochs=5)  # Training aware of quantization
wrapper.convert_to_ov(example_input=example_input)
```

**Why quantize?** Reduces model size by ~75% and speeds up inference, with minimal accuracy loss.

## Save & Load Organized Models

```python
# Save everything in a clean folder structure
model_dir = wrapper.save_ir_organized(
    base_path="./models", 
    model_name="my_classifier"
)

# Creates:
# ./models/my_classifier/
#   ├── my_classifier.xml  (model structure)
#   ├── my_classifier.bin  (model weights)
#   └── metadata.json      (training info, timestamps)

# Load later in production
production_wrapper = ClassificationWrapper(tf.keras.Sequential([tf.keras.layers.Dense(10)]))
production_wrapper.load_ir_from_folder(model_dir)
production_wrapper.setup_core()
production_wrapper.compile(device="CPU")

# Ready for fast inference!
result = production_wrapper.infer({0: new_image})
```

## Complete Inference Example

```python
import numpy as np

# Prepare test image (224x224 RGB)
test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Method 1: Synchronous (blocking)
result = wrapper.infer({0: test_image})
output_tensor = result[list(result.keys())[0]]
predicted_class = np.argmax(output_tensor, axis=1)[0]
confidence = np.max(output_tensor)

print(f"Predicted: Class {predicted_class} (confidence: {confidence:.3f})")

# Method 2: Asynchronous (non-blocking)
def on_inference_done(request, userdata):
    output = request.get_output_tensor(0).data
    predicted = np.argmax(output, axis=1)[0]
    print(f"Async result: Class {predicted}")

request = wrapper.infer({0: test_image}, async_mode=True, callback=on_inference_done)
# Do other work here...
request.wait()  # Wait for completion
```

## Task-Specific Wrappers

**Regression Example**
```python
from ov_training_kit.tensorflow import RegressionWrapper

regressor = RegressionWrapper(regression_model)
regressor.fit(x_train, y_train, epochs=10)

# Built-in regression metrics
mse = regressor.evaluate_mse(x_test, y_test)
mae = regressor.evaluate_mae(x_test, y_test)
r2_score = regressor.evaluate_r2(x_test, y_test)
print(f"MSE: {mse:.3f} | MAE: {mae:.3f} | R²: {r2_score:.3f}")
```

**Segmentation Example**
```python
from ov_training_kit.tensorflow import SegmentationWrapper

segmenter = SegmentationWrapper(segmentation_model)
segmenter.fit(x_train, y_train, epochs=10)

# Segmentation metrics
iou = segmenter.evaluate_iou(x_test, y_test, num_classes=21)
dice = segmenter.evaluate_dice(x_test, y_test, num_classes=21)
print(f"IoU: {iou:.3f} | Dice: {dice:.3f}")
```

## Performance Benchmarking

```python
# Test inference speed
test_input = {0: np.random.rand(8, 224, 224, 3).astype(np.float32)}

# Warm up
for _ in range(10):
    wrapper.infer(test_input)

# Benchmark 100 iterations
avg_time = wrapper.benchmark(test_input, num_iter=100)
throughput = 8 / avg_time  # batch_size / time_per_batch

print(f"Average time: {avg_time*1000:.1f}ms")
print(f"Throughput: {throughput:.1f} images/sec")

# Check if your CPU supports caching (faster startup)
caching_supported = wrapper.is_caching_supported("CPU")
print(f"Model caching supported: {caching_supported}")
```

## Requirements

- TensorFlow ≥ 2.12.0
- OpenVINO ≥ 2023.0  
- TensorFlow Model Optimization ≥ 0.7.0
- NumPy

---

**📈 Performance Gains**: Typical speedups of 2-5x on Intel CPUs compared to standard TensorFlow inference.

**🎯 Use Cases**: Perfect for deploying models on edge devices, servers, or laptops without GPUs.

## 🎓 Credits & License

Developed as part of a GSoC project with the OpenVINO community. Special thanks to mentors Shivam Basia and Aishwarye Omer for their guidance.

Licensed under Apache 2.0.