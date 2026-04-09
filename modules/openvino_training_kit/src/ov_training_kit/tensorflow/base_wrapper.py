try:
    import intel_extension_for_tensorflow as itex
    ITEX_AVAILABLE = True
    print("[OpenVINO] Intel Extension for TensorFlow (ITEX) loaded successfully")
except ImportError:
    ITEX_AVAILABLE = False
    itex = None
    print("[OpenVINO] Intel Extension for TensorFlow (ITEX) not available - using standard TensorFlow")

import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
class BaseWrapper:
    """
    High-level wrapper for TensorFlow → OpenVINO workflows.
    Supports training, evaluation, prediction, quantization, IR export, and benchmarking.
    All operations are optimized for Intel CPUs via ITEX if available.
    """

    def __init__(self, model):
        """
        Initialize with a tf.keras.Model.
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError("Model must be a tf.keras.Model")
        self.model = model
        self.ov_model = None
        self.compiled_model = None
        self.quantized = False
        print(f"[OpenVINO] Wrapper initialized with {type(model).__name__}")

    # =========================
    # Minimal code changes: TensorFlow-like API
    # =========================

    def fit(self, x, y, **kwargs):
        """
        Same signature as tf.keras.Model.fit.
        """
        return self.model.fit(x, y, **kwargs)

    def train(self, x, y, **kwargs):
        """
        Alias for fit, for compatibility.
        """
        return self.fit(x, y, **kwargs)

    def score(self, x, y, metric_fn=None, **kwargs):
        """
        Evaluate the model. Default metric: accuracy for classification, r2 for regression.
        """
        results = self.model.evaluate(x, y, **kwargs)
        if metric_fn is not None:
            preds = self.model.predict(x)
            return metric_fn(preds, y)
        # If model.evaluate returns accuracy, use it
        if isinstance(results, (list, tuple)) and len(results) > 1:
            return results[-1]
        return results

    def predict(self, x, **kwargs):
        """
        Same signature as tf.keras.Model.predict.
        """
        return self.model.predict(x, **kwargs)

    def save(self, filepath, **kwargs):
        """
        Save the model in TensorFlow format.
        """
        self.model.save(filepath, **kwargs)
        print(f"[OpenVINO] Model saved: {filepath}")

    def load(self, filepath, **kwargs):
        """
        Load a model from TensorFlow format.
        """
        self.model = tf.keras.models.load_model(filepath, **kwargs)
        print(f"[OpenVINO] Model loaded: {filepath}")

    def save_checkpoint(self, filepath, **kwargs):
        """
        Save model weights.
        """
        self.model.save_weights(filepath)
        print(f"[OpenVINO] Checkpoint (weights) saved: {filepath}")

    def load_checkpoint(self, filepath, **kwargs):
        """
        Load model weights from checkpoint.
        """
        self.model.load_weights(filepath)
        print(f"[OpenVINO] Checkpoint (weights) loaded: {filepath}")

    # =========================
    # Quantization (PTQ/QAT via TFMOT)
    # =========================

    def quantize(self, **kwargs):
        """
        Quantize the model using TensorFlow Model Optimization Toolkit (TFMOT).
        - If used AFTER training: applies Post-Training Quantization (PTQ)
        - If used BEFORE training: prepares for Quantization-Aware Training (QAT)
        """
        try:
            import tensorflow_model_optimization as tfmot
        except ImportError:
            raise ImportError("tensorflow_model_optimization (TFMOT) is required for quantization.")

        # Apply quantization (works for both PTQ and QAT preparation)
        quantize_model = tfmot.quantization.keras.quantize_model(self.model)
        self.model = quantize_model
        self.quantized = True
        print("[OpenVINO] Model quantized. Use after training for PTQ, or before training for QAT.")
        return self.model
    
    # =========================
    # Conversion & Export
    # =========================

    def convert_to_ov(self, example_input=None, input_shape=None, compress_to_fp16=True, **kwargs):
        """
        Convert the TensorFlow model to OpenVINO IR.
        """
        import openvino as ov
        if example_input is not None:
            input_shape = example_input.shape
        elif input_shape is None:
            raise ValueError("Provide example_input or input_shape for conversion.")
        self.ov_model = ov.convert_model(self.model, example_input=example_input, input=input_shape, **kwargs)
        print("[OpenVINO] Model converted to OpenVINO IR.")
        return self.ov_model

    def save_ir(self, xml_path, compress_to_fp16=True):
        """
        Save the OpenVINO IR model to disk.
        """
        import openvino as ov
        if self.ov_model is None:
            raise RuntimeError("No OpenVINO model to save. Run convert_to_ov first.")
        ov.save_model(self.ov_model, xml_path, compress_to_fp16=compress_to_fp16)
        print(f"[OpenVINO] IR saved: {xml_path}")

    def save_ir_organized(self, base_path, model_name="model", compress_to_fp16=True, include_metadata=True):
        """
        Save the OpenVINO IR model in an organized folder structure.
        """
        import openvino as ov
        model_dir = os.path.join(base_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        xml_path = os.path.join(model_dir, f"{model_name}.xml")
        ov.save_model(self.ov_model, xml_path, compress_to_fp16=compress_to_fp16)
        if include_metadata:
            metadata = {
                'model_name': model_name,
                'created_at': datetime.now().isoformat(),
                'quantized': self.quantized,
                'tensorflow_info': {
                    'model_class': type(self.model).__name__
                }
            }
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        print(f"[OpenVINO] IR saved to organized folder: {model_dir}")
        return model_dir

    def load_ir_from_folder(self, model_dir, model_name=None):
        """
        Load OpenVINO IR model from an organized folder.
        """
        import openvino as ov
        if model_name is None:
            xml_files = [f for f in os.listdir(model_dir) if f.endswith('.xml')]
            if not xml_files:
                raise FileNotFoundError(f"No .xml files found in {model_dir}")
            if len(xml_files) > 1:
                raise ValueError(f"Multiple .xml files found in {model_dir}. Specify model_name.")
            model_name = xml_files[0][:-4]
        xml_path = os.path.join(model_dir, f"{model_name}.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        self.ov_model = ov.Core().read_model(xml_path)
        print(f"[OpenVINO] IR loaded from folder: {xml_path}")
        return xml_path

    # =========================
    # OpenVINO Core & Compilation
    # =========================

    def setup_core(self, cache_dir=None, mmap=True):
        """
        Create and configure the OpenVINO Core object.
        """
        import openvino as ov
        import openvino.properties as props
        self.core = ov.Core()
        config = {}
        if cache_dir:
            config[props.cache_dir] = cache_dir
        if mmap:
            config[props.enable_mmap] = True
        if config:
            self.core.set_property(config)
        print(f"[OpenVINO] Core initialized with config: {config}")

    def compile(self, device="CPU", config=None, model_path=None):
        """
        Compile the OpenVINO model for inference.
        """
        import openvino as ov
        if not hasattr(self, "core") or self.core is None:
            self.core = ov.Core()
        if model_path:
            model = self.core.read_model(model_path)
        else:
            if self.ov_model is None:
                raise RuntimeError("No OpenVINO model to compile. Run convert_to_ov first.")
            model = self.ov_model
        self.compiled_model = self.core.compile_model(model, device_name=device, config=config or {})
        print(f"[OpenVINO] Model compiled for device: {device}")

    # =========================
    # Inference & Benchmark
    # =========================

    def infer(self, input_data, async_mode=False, callback=None):
        """
        Run inference with the compiled OpenVINO model.
        """
        if self.compiled_model is None:
            raise RuntimeError("No compiled model. Run compile first.")
        if not async_mode:
            result = self.compiled_model(input_data)
            return result
        else:
            request = self.compiled_model.create_infer_request()
            if callback:
                def safe_callback(*args):
                    if len(args) == 2:
                        try:
                            callback(args[0], args[1])
                        except TypeError:
                            callback(args[0])
                    elif len(args) == 1:
                        try:
                            callback(args[0], None)
                        except TypeError:
                            callback(args[0])
                request.set_callback(safe_callback, None)
            request.start_async(input_data)
            return request

    def benchmark(self, input_data, num_iter=100):
        """
        Simple benchmarking of the compiled OpenVINO model (synchronous).
        Returns average inference time in seconds.
        """
        import time
        if self.compiled_model is None:
            raise RuntimeError("No compiled model. Run compile first.")
        times = []
        for _ in range(num_iter):
            start = time.time()
            _ = self.compiled_model(input_data)
            times.append(time.time() - start)
        avg_time = sum(times) / len(times)
        print(f"[OpenVINO] Average inference time: {avg_time:.4f}s")
        return avg_time

    # =========================
    # Utilities
    # =========================

    @staticmethod
    def make_tf_dataset(x, y=None, batch_size=32, shuffle=True):
        """
        Utility to create a tf.data.Dataset from numpy arrays.
        """
        if y is not None:
            ds = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            ds = tf.data.Dataset.from_tensor_slices(x)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x))
        ds = ds.batch(batch_size)
        return ds

    @staticmethod
    def is_caching_supported(device="CPU"):
        """
        Check if the device supports model caching (OpenVINO).
        """
        import openvino as ov
        import openvino.properties.device as device_props
        core = ov.Core()
        return 'EXPORT_IMPORT' in core.get_property(device, device_props.capabilities)

    @staticmethod
    def optimal_num_requests(compiled_model):
        """Get optimal number of inference requests."""
        try:
            # Tentar nova API primeiro
            return compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        except:
            try:
                # Fallback para API antiga
                import openvino.properties as props
                return compiled_model.get_property(props.optimal_number_of_infer_requests)
            except:
                # Fallback final
                return 1