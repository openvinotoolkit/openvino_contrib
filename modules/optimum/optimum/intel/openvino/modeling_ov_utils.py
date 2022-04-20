# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import errno
import logging
import shutil
import time

import numpy as np

try:
    from openvino.runtime import Core

    is_openvino_api_2 = True
except ImportError:
    from openvino.inference_engine import IECore as Core

    is_openvino_api_2 = False

from transformers.file_utils import cached_path, hf_bucket_url
from transformers.file_utils import is_torch_available
from transformers import (
    TF2_WEIGHTS_NAME,
    AutoConfig,
)


if is_torch_available():
    import torch
    from transformers.generation_utils import GenerationMixin
    from transformers.file_utils import ModelOutput
    from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput
else:
    from collections import namedtuple

    class GenerationMixin(object):
        def __init__(self):
            pass

    QuestionAnsweringModelOutput = namedtuple("QuestionAnsweringModelOutput", ["start_logits", "end_logits"])
    ModelOutput = namedtuple("ModelOutput", ["logits"])

logger = logging.getLogger(__name__)

OV_WEIGHTS_NAME = "ov_model.xml"
ie = Core()


def load_ov_model_from_pytorch(model, inputs=None):
    import io

    buf = io.BytesIO()
    if inputs is None:
        dummy_input_ids = torch.zeros((1, 18), dtype=torch.int32)
        dummy_mask = torch.zeros((1, 18), dtype=torch.int32)
        if model.config.model_type == "gpt2":
            if model.config.use_cache:
                raise NotImplementedError("GPT2 model with use_cache=True is not implemented for OpenVINO backend")

            inputs = (dummy_input_ids, None, dummy_mask)
        elif model.main_input_name == "input_values":
            inputs = torch.zeros((1, 16000), dtype=torch.float32)
        else:
            inputs = (dummy_input_ids, dummy_mask)

        input_names = [model.main_input_name, "attention_mask"]
    else:
        input_names = []
        for name, tensor in inputs.items():
            if tensor is None:
                continue
            if name == "past_key_values":
                for i in range(len(tensor)):
                    for j in range(len(tensor[i])):
                        input_names.append(f"past_key_values.{i}.{j}")
            else:
                input_names.append(name)

        inputs = tuple(inputs.values())

    if model.__class__.__name__.endswith("ForQuestionAnswering"):
        outputs = ["output_s", "output_e"]
    else:
        outputs = ["output"]

    with torch.no_grad():
        # Estimate model size. If it larger than 2GB - protobuf will fail export to ONNX.
        mem_size = np.sum([t.element_size() * np.prod(t.shape) * 1e-6 for t in model.state_dict().values()])

        use_external_data_format = mem_size > 2000

        # TODO: create "model" folder in cache
        if use_external_data_format:
            model_cache_dir = f"openvino_model_cache_{time.time()}"
            os.makedirs(model_cache_dir)

        torch.onnx.export(
            model,
            inputs,
            buf if not use_external_data_format else os.path.join(model_cache_dir, "model.onnx"),
            input_names=input_names,
            output_names=outputs,
            opset_version=11,
            use_external_data_format=use_external_data_format,
        )

    if use_external_data_format:
        if is_openvino_api_2:
            net = ie.read_model(os.path.join(model_cache_dir, "model.onnx"))
        else:
            net = ie.read_network(os.path.join(model_cache_dir, "model.onnx"))

        try:
            shutil.rmtree(model_cache_dir)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
    else:
        if is_openvino_api_2:
            net = ie.read_model(buf.getvalue(), b"")
        else:
            net = ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
    return OVPreTrainedModel(net, model.config)


def load_ov_model_from_tf(model, tf_weights_path):
    import subprocess  # nosec

    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    func = tf.function(lambda input_ids, attention_mask: model(input_ids, attention_mask=attention_mask))
    func = func.get_concrete_function(
        input_ids=tf.TensorSpec((None, None), tf.int32, name="input_ids"),
        attention_mask=tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
    )
    if isinstance(func.structured_outputs, tuple):
        output_names = [out.name for out in func.structured_outputs]
    else:
        output_names = [out.name for out in func.structured_outputs.values()]

    frozen_func = convert_variables_to_constants_v2(func)
    graph_def = frozen_func.graph.as_graph_def()

    cache_dir = os.path.dirname(tf_weights_path)
    pb_model_path = os.path.join(cache_dir, "frozen_graph.pb")
    with tf.io.gfile.GFile(pb_model_path, "wb") as f:
        f.write(graph_def.SerializeToString())

    subprocess.run(  # nosec
        [
            "mo",
            "--output_dir",
            cache_dir,
            "--input_model",
            pb_model_path,
            "--model_name",
            os.path.basename(tf_weights_path),
            "--input",
            "input_ids,attention_mask",
            "--output",
            ",".join(output_names),
            "--input_shape",
            "[1, 11], [1, 11]",
            "--disable_nhwc_to_nchw",
        ],
        check=True,
    )

    try:
        os.remove(pb_model_path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

    if is_openvino_api_2:
        net = ie.read_model(tf_weights_path + ".xml")
    else:
        net = ie.read_network(tf_weights_path + ".xml")
    return OVPreTrainedModel(net, model.config)


def load_ov_model_from_ir(xml_path, bin_path, config):
    if not xml_path.endswith(".xml"):
        import shutil

        shutil.copyfile(xml_path, xml_path + ".xml")
        xml_path += ".xml"

    if is_openvino_api_2:
        net = ie.read_model(xml_path, bin_path)
    else:
        net = ie.read_network(xml_path, bin_path)
    return OVPreTrainedModel(net, config)


def load_model_from_cache(model_name_or_path, model_arch, cache_dir, filename, config):
    url = hf_bucket_url(model_name_or_path, filename=filename)
    path = cached_path(url, cache_dir=cache_dir) + "." + model_arch
    xml_path = path + ".xml"
    bin_path = path + ".bin"
    model = None
    if os.path.exists(xml_path) and os.path.exists(bin_path):
        logger.info(f"Load OpenVINO model from cache: {xml_path}")
        model = load_ov_model_from_ir(xml_path, bin_path, config)
    return model, path


class OVPreTrainedModel(GenerationMixin):
    _pt_auto_model = None
    _tf_auto_model = None

    def __init__(self, net, config):
        super().__init__()
        self.net = net

        if is_openvino_api_2:
            # Workaround for a bug with "input_ids:0" name
            for inp in self.net.inputs:
                name = inp.get_any_name().split(":")[0]
                inp.get_tensor().set_names(set([name]))
            self.input_names = [inp.get_any_name() for inp in self.net.inputs]
            self.output_names = [out.get_any_name() for out in self.net.outputs]
        else:
            self.input_names = [inp for inp in self.net.input_info]
            self.output_names = [out for out in self.net.outputs]
        self.exec_net = None
        self.config = config
        self.max_length = 0
        self.ov_config = {"PERFORMANCE_HINT": "LATENCY"} if is_openvino_api_2 else {}
        self.ov_device = "CPU"
        self.use_dynamic_shapes = is_openvino_api_2

        self.main_input_name = None
        for name in ["input_ids", "input_values", "decoder_input_ids"]:
            if name in self.input_names:
                self.main_input_name = name
        if self.main_input_name is None:
            raise Exception(f"Cannot determine main_input_name from {self.input_names}")

        if is_torch_available():
            self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        cache_dir = kwargs.get("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        from_tf = kwargs.pop("from_tf", False)
        from_ov = kwargs.get("from_ov", not (from_pt | from_tf))
        force_download = kwargs.get("force_download", False)
        resume_download = kwargs.get("resume_download", False)
        proxies = kwargs.get("proxies", None)
        local_files_only = kwargs.get("local_files_only", False)
        use_auth_token = kwargs.get("use_auth_token", None)
        revision = kwargs.get("revision", None)
        from_pipeline = kwargs.get("_from_pipeline", None)
        from_auto_class = kwargs.get("_from_auto", False)

        config = kwargs.get("config") if "config" in kwargs else AutoConfig.from_pretrained(model_name_or_path)

        if from_pt:
            model = cls._pt_auto_model.from_pretrained(model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_pytorch(model)
        elif from_tf:
            model, cache_path = load_model_from_cache(
                model_name_or_path, cls.__name__, cache_dir, TF2_WEIGHTS_NAME, config
            )
            if model is not None:
                return model
            model = cls._tf_auto_model.from_pretrained(model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_tf(model, cache_path)

        user_agent = {"file_type": "model", "framework": "openvino", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load model
        OV_BIN_NAME = OV_WEIGHTS_NAME.replace(".xml", ".bin")
        if model_name_or_path is not None:
            if os.path.isdir(model_name_or_path):
                if (
                    from_ov
                    and os.path.isfile(os.path.join(model_name_or_path, OV_WEIGHTS_NAME))
                    and os.path.isfile(os.path.join(model_name_or_path, OV_BIN_NAME))
                ):
                    # Load from an OpenVINO IR
                    archive_files = [os.path.join(model_name_or_path, name) for name in [OV_WEIGHTS_NAME, OV_BIN_NAME]]
                else:
                    raise EnvironmentError(
                        f"Error no files named {[OV_WEIGHTS_NAME, OV_BIN_NAME]} found in directory "
                        f"{model_name_or_path} or `from_ov` set to False"
                    )
            # elif os.path.isfile(model_name_or_path) or is_remote_url(model_name_or_path):
            #     archive_file = model_name_or_path
            else:
                names = [OV_WEIGHTS_NAME, OV_BIN_NAME]
                archive_files = [
                    hf_bucket_url(
                        model_name_or_path,
                        filename=name,
                        revision=revision,
                    )
                    for name in names
                ]

            # redirect to the cache, if necessary
            try:
                resolved_archive_files = [
                    cached_path(
                        archive_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
                    for archive_file in archive_files
                ]
            except EnvironmentError as err:
                logger.error(err)
                name = model_name_or_path
                msg = (
                    f"Can't load weights for '{name}'. Make sure that:\n\n"
                    f"- '{name}' is a correct model identifier listed on 'https://huggingface.co/models'\n"
                    f"  (make sure '{name}' is not a path to a local directory with something else, in that case)\n\n"
                    f"- or '{name}' is the correct path to a directory containing a file named {OV_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_files == archive_files:
                logger.info(f"loading weights file {archive_files}")
            else:
                logger.info(f"loading weights file {archive_files} from cache at {resolved_archive_files}")
        else:
            resolved_archive_files = None

        return load_ov_model_from_ir(*resolved_archive_files, config=config)

    def save_pretrained(
        self,
        save_directory,
        **kwargs,
    ):
        """
        Save model in OpenVINO IR format into a directory
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        xml_path = os.path.join(save_directory, OV_WEIGHTS_NAME)
        if is_openvino_api_2:
            import openvino.runtime.passes as passes

            pass_manager = passes.Manager()
            pass_manager.register_pass("Serialize", xml_path, xml_path.replace(".xml", ".bin"))
            pass_manager.run_passes(self.net)
        else:
            self.net.serialize(xml_path)

    def to(self, device):
        self.ov_device = device

    def set_config(self, config):
        self.ov_config = config

    def _load_network(self):
        if is_openvino_api_2:
            if self.use_dynamic_shapes:
                shapes = {}
                for inp in self.net.inputs:
                    shapes[inp] = inp.get_partial_shape()
                    shapes[inp][0] = -1
                    if inp.get_any_name().startswith("past_key_values"):
                        shapes[inp][2] = -1
                    else:
                        shapes[inp][1] = -1

                self.net.reshape(shapes)
            compiled_model = ie.compile_model(self.net, self.ov_device, self.ov_config)
            self.exec_net = compiled_model.create_infer_request()
        else:
            self.exec_net = ie.load_network(self.net, self.ov_device, self.ov_config)

    def _process_data_api_2021(self, inputs):
        # In case of batching, we process samples one by one instead of
        # single forward pass. It is done because of heavy load_network step.
        batch_size = inputs[self.main_input_name].shape[0]
        if batch_size > 1:
            outs = {k: np.zeros([batch_size] + out.shape[1:], np.float32) for k, out in self.net.outputs.items()}
            for i in range(batch_size):
                outs_i = self.exec_net.infer({name: inp[i : i + 1] for name, inp in inputs.items()})
                for name in outs:
                    # OpenVINO produces redundant output for Stack layers. Ignore them
                    if name.endswith("/stack"):
                        continue
                    outs[name][i] = outs_i[name]
        else:
            outs = self.exec_net.infer(inputs)
        return outs

    def _process_data_api_2022(self, inputs):
        outs = self.exec_net.infer(inputs)
        outs = {out.get_any_name(): value for out, value in outs.items()}
        return outs

    def _prepare_nlp_inputs(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        use_cache=False,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids) if attention_mask is None else attention_mask,
        }

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids) if token_type_ids is None else token_type_ids

        return inputs

    def _prepare_audio_inputs(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return {"input_values": input_values}

    def _process_data(self, inputs, return_dict):
        inp_length = inputs[self.main_input_name].shape[1]

        # If <max_length> specified, pad inputs by zeros
        if inp_length < self.max_length:
            for name in inputs:
                shape = inputs[name].shape
                if shape[1] != self.max_length:
                    pad = np.zeros([len(shape), 2], dtype=np.int32)
                    pad[1, 1] = self.max_length - shape[1]
                    inputs[name] = np.pad(inputs[name], pad)

        # OpenVINO >= 2022.1 supports dynamic shapes input.
        if not is_openvino_api_2:
            inputs_info = self.net.input_info
            input_ids = inputs[self.main_input_name]
            if inputs_info[self.main_input_name].input_data.shape[1] != input_ids.shape[1]:
                # Use batch size 1 because we process batch sequently.
                shapes = {key: [1] + list(inputs[key].shape[1:]) for key in inputs_info}
                logger.info(f"Reshape model to {shapes}")
                self.net.reshape(shapes)
                self.exec_net = None
        elif is_openvino_api_2 and not self.use_dynamic_shapes:
            # TODO
            pass

        if self.exec_net is None:
            self._load_network()

        if is_openvino_api_2:
            outs = self._process_data_api_2022(inputs)
        else:
            outs = self._process_data_api_2021(inputs)

        logits = outs["output"] if "output" in outs else next(iter(outs.values()))

        past_key_values = None
        if self.config.architectures[0].endswith("ForConditionalGeneration") and self.config.use_cache:
            past_key_values = [[]]
            for name in outs:
                if name == "output":
                    continue
                if len(past_key_values[-1]) == 4:
                    past_key_values.append([])
                past_key_values[-1].append(torch.tensor(outs[name]))

            past_key_values = tuple([tuple(val) for val in past_key_values])

        # Trunc padded values
        if inp_length != logits.shape[1]:
            logits = logits[:, :inp_length]

        if not return_dict:
            return [logits]

        arch = self.config.architectures[0]
        if arch.endswith("ForSequenceClassification"):
            return SequenceClassifierOutput(logits=logits)
        elif arch.endswith("ForQuestionAnswering"):
            return QuestionAnsweringModelOutput(start_logits=outs["output_s"], end_logits=outs["output_e"])
        else:
            return ModelOutput(logits=torch.tensor(logits), past_key_values=past_key_values)

    def forward(self, *args, **kwargs):
        if self.main_input_name in ["input_ids", "decoder_input_ids"]:
            inputs = self._prepare_nlp_inputs(*args, **kwargs)
        elif self.main_input_name == "input_values":
            inputs = self._prepare_audio_inputs(*args, **kwargs)
        else:
            raise Exception(f"Unexpected main_input_name: {self.main_input_name}")

        if "return_dict" in kwargs:
            return_dict = kwargs["return_dict"]
        else:
            return_dict = self.config.use_return_dict if hasattr(self.config, "use_return_dict") else None

        return self._process_data(inputs, return_dict)

    def generate(self, input_ids, *args, **kwargs):
        if not is_torch_available():
            raise Exception("PyTorch is required to run generators")

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)

        # OpenVINO >= 2022.1 supports dynamic inputs so max_length is optional.
        if not self.use_dynamic_shapes:
            max_length = kwargs.get("max_length", None)
            self.max_length = max_length if max_length is not None else self.config.max_length
            self.max_length -= 1

        return super().generate(input_ids, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
