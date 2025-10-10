import tensorflow as tf
import time
import numpy as np
import os
import psutil
import sys

from openvino_kit.tensorflow import ClassificationWrapper

def measure_tensorflow_inference(model, input_tensor, num_iter=100):
    times = []
    mem_usages = []
    for _ in range(num_iter):
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        start = time.time()
        _ = model(input_tensor)
        times.append(time.time() - start)
        end_mem = psutil.Process(os.getpid()).memory_info().rss
        mem_usages.append(end_mem - start_mem)
    avg_time = sum(times) / len(times)
    avg_mem = sum(mem_usages) / len(mem_usages)
    return avg_time, avg_mem

def measure_openvino_inference(wrapper, input_array, num_iter=100):
    times = []
    mem_usages = []
    for _ in range(num_iter):
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        start = time.time()
        _ = wrapper.infer({0: input_array})
        times.append(time.time() - start)
        end_mem = psutil.Process(os.getpid()).memory_info().rss
        mem_usages.append(end_mem - start_mem)
    avg_time = sum(times) / len(times)
    avg_mem = sum(mem_usages) / len(mem_usages)
    return avg_time, avg_mem

def get_model_size(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)  # MB

if __name__ == "__main__":
    # 1. TensorFlow baseline
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    x_train = np.random.randn(1000, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, (1000,))

    # Measure TensorFlow training time
    start_train_tf = time.time()
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
    end_train_tf = time.time()
    train_time_tf = end_train_tf - start_train_tf
    print(f"[TensorFlow] Training time: {train_time_tf:.2f}s")

    input_tensor = np.random.randn(1, 32, 32, 3).astype(np.float32)
    avg_time_tf, avg_mem_tf = measure_tensorflow_inference(model, input_tensor, num_iter=100)
    print(f"[TensorFlow] Avg inference time: {avg_time_tf:.4f}s | Avg memory usage: {avg_mem_tf/1024/1024:.2f} MB")

    # 2. OpenVINO quantized
    wrapper = ClassificationWrapper(tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]))
    wrapper.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Measure training time with wrapper
    start_train = time.time()
    wrapper.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
    end_train = time.time()
    train_time = end_train - start_train
    print(f"[OpenVINO Wrapper] Training time (TensorFlow): {train_time:.2f}s")
    
    try:
        wrapper.quantize()
        example_input = np.random.randn(1, 32, 32, 3).astype(np.float32)
        wrapper.convert_to_ov(example_input=example_input)
        wrapper.save_ir_organized(
            base_path="./my_exported_models_tf",
            model_name="cnn_quantized",
            compress_to_fp16=True,
            include_metadata=True
        )
        wrapper.setup_core(cache_dir="./ov_cache_tf", mmap=True)
        wrapper.compile(device="CPU")
        input_array = np.random.randn(1, 32, 32, 3).astype(np.float32)
        avg_time_ov, avg_mem_ov = measure_openvino_inference(wrapper, input_array, num_iter=100)
        print(f"[OpenVINO] Avg inference time: {avg_time_ov:.4f}s | Avg memory usage: {avg_mem_ov/1024/1024:.2f} MB")
    except Exception as e:
        print(f"[OpenVINO] Pipeline failed/skipped: {e}")

    # 3. Model size comparison
    tf_params = model.count_params()
    tf_size = tf_params * 4 / (1024 * 1024)  # Assuming float32
    ov_bin_path = "./my_exported_models_tf/cnn_quantized/cnn_quantized.bin"
    ov_size = get_model_size(ov_bin_path) if os.path.exists(ov_bin_path) else 0
    print(f"[TensorFlow] Model size: {tf_size:.2f} MB")
    print(f"[OpenVINO] Model size: {ov_size:.2f} MB")