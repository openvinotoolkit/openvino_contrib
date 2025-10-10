import tensorflow as tf
import numpy as np
import sys
import os

from openvino_kit.tensorflow import ClassificationWrapper

# 1. Load a TensorFlow model (simple CNN for CIFAR-10)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

wrapper = ClassificationWrapper(model)

# 2. Prepare real data (CIFAR-10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

# Use subset for quick testing
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:200]
y_test = y_test[:200]

# 3. Train the model
wrapper.fit(x_train, y_train, epochs=2, batch_size=32, 
           validation_data=(x_test, y_test), verbose=1)

# 4. Evaluate with metrics
acc = wrapper.score(x_test, y_test, batch_size=32)
print(f"Accuracy: {acc:.3f}")

# 5. Quantize (PTQ) after training
try:
    wrapper.quantize()
    print("✅ Model quantized successfully")
except Exception as e:
    print("Quantization skipped (TFMOT not installed or not supported):", e)

# 6. Convert to OpenVINO IR
example_input = np.random.randn(1, 32, 32, 3).astype(np.float32)
try:
    wrapper.convert_to_ov(example_input=example_input)
    print("✅ OpenVINO conversion successful")
except Exception as e:
    print("OpenVINO conversion skipped:", e)

# 7. Export IR model to organized folder
try:
    wrapper.save_ir_organized(
        base_path="./my_exported_models_tf",
        model_name="cnn_quantized",
        compress_to_fp16=True,
        include_metadata=True
    )
    print("✅ IR export successful")
except Exception as e:
    print("IR export skipped:", e)

# 8. Compile and run inference
try:
    wrapper.setup_core(cache_dir="./ov_cache_tf", mmap=True)
    wrapper.compile(device="CPU")
    print("✅ OpenVINO compile successful")
except Exception as e:
    print("OpenVINO compile skipped:", e)

# 9. Inference on new data
try:
    # Use a real image from the test set
    img = x_test[0:1]  # Shape: (1, 32, 32, 3)
    label = y_test[0]
    result = wrapper.infer({0: img})
    pred_class = int(np.argmax(list(result.values())[0]))
    print(f"Inference OK! Predicted class: {pred_class}, True label: {label}")
except Exception as e:
    print("OpenVINO inference not performed:", e)