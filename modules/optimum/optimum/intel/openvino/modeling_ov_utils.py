# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import errno
import logging
import shutil
import time

import numpy as np

from transformers.file_utils import cached_path, hf_bucket_url
from transformers.file_utils import is_torch_available
from transformers import TF2_WEIGHTS_NAME, AutoConfig, AutoTokenizer

from openvino.runtime import Core
from openvino.model_zoo.model_api.adapters.openvino_adapter import OpenvinoAdapter
from openvino.model_zoo.model_api.adapters.ovms_adapter import OVMSAdapter


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


def load_ov_model_from_pytorch(model, inputs=None, ov_config=None):
    import io

    buf = io.BytesIO()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        input_names = tokenizer.model_input_names
    except OSError:
        input_names = [model.main_input_name, "attention_mask"]

    if "token_type_ids" in input_names:
        # token_type_ids should be last input
        input_names = [model.main_input_name, "attention_mask", "token_type_ids"]

    if inputs is None:
        dummy_inputs = torch.zeros((1, 18), dtype=torch.int64)

        if model.config.model_type == "gpt2":
            if model.config.use_cache:
                raise NotImplementedError("GPT2 model with use_cache=True is not implemented for OpenVINO backend")

            inputs = (dummy_inputs, None, dummy_inputs)
        elif model.main_input_name == "input_values":
            inputs = torch.zeros((1, 16000), dtype=torch.float32)
        else:
            inputs = tuple([dummy_inputs] * len(input_names))
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
            opset_version=12,
            use_external_data_format=use_external_data_format,
        )

    if use_external_data_format:
        inference_adapter = OpenvinoAdapter(ie, (os.path.join(model_cache_dir, "model.onnx")), plugin_config=ov_config)

        try:
            shutil.rmtree(model_cache_dir)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
    else:
        inference_adapter = OpenvinoAdapter(ie, buf.getvalue(), b"", plugin_config=ov_config)
    return OVPreTrainedModel(inference_adapter, ov_config, model.config)


def load_ov_model_from_tf(model, tf_weights_path, ov_config=None):
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

    # TODO: fix for models with different input names
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

    inference_adapter = OpenvinoAdapter(ie, tf_weights_path + ".xml", plugin_config=ov_config)
    return OVPreTrainedModel(inference_adapter, ov_config, model.config)


def load_ov_model_from_ir(xml_path, bin_path, config, ov_config=None):
    if not xml_path.endswith(".xml"):
        import shutil

        shutil.copyfile(xml_path, xml_path + ".xml")
        xml_path += ".xml"

    inference_adapter = OpenvinoAdapter(ie, xml_path, bin_path, plugin_config=ov_config)
    return OVPreTrainedModel(inference_adapter, ov_config, config)


def load_model_from_cache(model_name_or_path, model_arch, cache_dir, filename, config, ov_config=None):
    url = hf_bucket_url(model_name_or_path, filename=filename)
    path = cached_path(url, cache_dir=cache_dir) + "." + model_arch
    xml_path = path + ".xml"
    bin_path = path + ".bin"
    model = None
    if os.path.exists(xml_path) and os.path.exists(bin_path):
        logger.info(f"Load OpenVINO model from cache: {xml_path}")
        model = load_ov_model_from_ir(xml_path, bin_path, config, ov_config)
    return model, path


class OVPreTrainedModel(GenerationMixin):
    _pt_auto_model = None
    _tf_auto_model = None

    def __init__(self, inference_adapter, ov_config, config):
        super().__init__()
        self.inference_adapter = inference_adapter
        self.ov_config = ov_config
        self.model_initialized = False

        # Workaround for a bug with "input_ids:0" name
        # for inp in self.net.inputs:
        #    name = inp.get_any_name().split(":")[0]
        #    inp.get_tensor().set_names(set([name]))

        inputs = inference_adapter.get_input_layers()
        outputs = inference_adapter.get_output_layers()

        self.input_names = list(inputs.keys())
        self.output_names = list(outputs.keys())

        # self.exec_net = None
        self.config = config
        self.max_length = 0
        # self.ov_config = {"PERFORMANCE_HINT": "LATENCY"}
        # self.ov_device = "CPU"
        self.use_dynamic_shapes = True

        self.main_input_name = None
        for name in ["input_ids", "input_ids:0", "input_values", "decoder_input_ids"]:
            if name in self.input_names:
                self.main_input_name = name
        if self.main_input_name is None:
            raise Exception(f"Cannot determine main_input_name from {self.input_names}")

        if is_torch_available():
            self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        inference_backend = kwargs.get("inference_backend", "openvino")
        ov_config = kwargs.get("ov_config", {"PERFORMANCE_HINT": "LATENCY"})
        if inference_backend == "ovms":
            inference_adapter = OVMSAdapter(model_name_or_path)
            config = kwargs.get("config")
            if config is None:
                raise Exception("Config is required when using OVMS as a backend")
            return OVPreTrainedModel(inference_adapter, ov_config, config)

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
            return load_ov_model_from_pytorch(model, ov_config=ov_config)
        elif from_tf:
            model, cache_path = load_model_from_cache(
                model_name_or_path, cls.__name__, cache_dir, TF2_WEIGHTS_NAME, config, ov_config=ov_config
            )
            if model is not None:
                return model
            model = cls._tf_auto_model.from_pretrained(model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_tf(model, cache_path, ov_config=ov_config)

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
                archive_files = [hf_bucket_url(model_name_or_path, filename=name, revision=revision) for name in names]

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

        return load_ov_model_from_ir(*resolved_archive_files, config=config, ov_config=ov_config)

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save model in OpenVINO IR format into a directory
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        xml_path = os.path.join(save_directory, OV_WEIGHTS_NAME)
        import openvino.runtime.passes as passes

        pass_manager = passes.Manager()
        pass_manager.register_pass("Serialize", xml_path, xml_path.replace(".xml", ".bin"))
        # TO DO: disable saving for OVMSAdapter as model is not in place
        pass_manager.run_passes(self.inference_adapter.model)

    def to(self, device):
        self.ov_device = device

    def set_config(self, config):
        self.ov_config = config

    def _load_network(self):
        if self.use_dynamic_shapes and not isinstance(self.inference_adapter, OVMSAdapter):
            shapes = {}
            for input_name, metadata in self.inference_adapter.get_input_layers().items():
                shapes[input_name] = list(metadata.shape)
                shapes[input_name][0] = -1
                if input_name.startswith("past_key_values"):
                    shapes[input_name][2] = -1
                else:
                    shapes[input_name][1] = -1

            self.inference_adapter.reshape_model(shapes)
        self.inference_adapter.load_model()
        self.model_initialized = True

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

        if not self.use_dynamic_shapes:
            # TODO
            pass

        if not self.model_initialized:
            self._load_network()

        # ovmsclient not supporting torch.Tensor input type - needed conversion to numpy
        inputs = {
            input_name: data.numpy() if type(data) is torch.Tensor else data for input_name, data in inputs.items()
        }

        outs = self.inference_adapter.infer_sync(inputs)
        logits = outs["output"] if "output" in outs else next(iter(outs.values()))

        past_key_values = None
        if self.config.architectures[0].endswith("ForConditionalGeneration") and self.config.use_cache:

            # OVMSAdapter does not guarantee output keys order
            # For use cases where such order is required we use workaround with output names mapping
            # OV model output names are mapped to numers and below we sort them to restore original order
            if type(self.inference_adapter) is OVMSAdapter:
                outs = dict(sorted(outs.items(), key=lambda item: int(item[0]) if item[0] != "output" else -1))

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
            # For OpenVINO API 1.0, output names are not necessarily in correct order
            if "output_s" in outs.keys() and "output_e" in outs.keys():
                output_s, output_e = "output_s", "output_e"
            else:
                # Get output names from model.
                # For quantized models, output names are not necessarily "output_s" and "output_e"
                output_s, output_e = list(outs.keys())

            return QuestionAnsweringModelOutput(start_logits=outs[output_s], end_logits=outs[output_e])
        else:
            return ModelOutput(logits=torch.tensor(logits), past_key_values=past_key_values)

    def forward(self, *args, **kwargs):
        if self.main_input_name in ["input_ids", "input_ids:0", "decoder_input_ids"]:
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

    # Experimental

    def create_ovms_image(self, image_tag):
        import docker
        import json
        import shutil

        # Prepare configuration file
        if not self.model_initialized:
            self._load_network()

        self.save_pretrained("/tmp/optimum/")

        model_configuration = {"name": "model", "base_path": "/opt/model"}
        if self.ov_config:
            model_configuration["plugin_config"] = self.ov_config

        config = {}
        config["model_config_list"] = [{"config": model_configuration}]

        with open("/tmp/optimum/config.json", "w") as outfile:
            json.dump(config, outfile)

        dockerfile_content = [
            "FROM openvino/model_server:latest\n",
            "COPY *.xml *.bin /opt/model/1/\n",
            "COPY config.json /opt/config.json\n",
            'ENTRYPOINT ["/ovms/bin/ovms", "--config_path", "/opt/config.json"]\n',
        ]

        with open("/tmp/optimum/Dockerfile", "w") as f:
            f.writelines(dockerfile_content)

        client = docker.from_env()
        client.images.build(path="/tmp/optimum", tag=image_tag)
        shutil.rmtree("/tmp/optimum/")
