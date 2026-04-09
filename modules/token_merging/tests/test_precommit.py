# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
import os

import numpy as np
from PIL import Image
import torch
import openvino.runtime as ov
from openvino import convert_model

import tomeov
from diffusers import StableDiffusionPipeline, DDPMScheduler
from optimum.intel.openvino import OVStableDiffusionPipeline
from optimum.exporters.openvino import export_from_model
import open_clip
import timm


class TokenMergingIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OV_DIFFUSION_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"
        self.OPENCLIP_MODEL = ("ViT-B-32", "laion400m_e32")
        self.TIMM_MODEL = "vit_tiny_patch16_224"
        
    def test_stable_diffusion(self):
        loaded_pipeline = StableDiffusionPipeline.from_pretrained(self.OV_DIFFUSION_MODEL_ID)
        prompt = "sailing ship in storm by Leonardo da Vinci"
        height = 128
        width = 128

        tomeov.patch_stable_diffusion(loaded_pipeline, ratio=0.3)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            export_from_model(loaded_pipeline, tmpdirname)
            ov_pipe = OVStableDiffusionPipeline.from_pretrained(tmpdirname, compile=False)
            ov_pipe.reshape(batch_size=1, height=height, width=width, num_images_per_prompt=1)
            ov_pipe.compile()
            ov_pipe(prompt, num_inference_steps=1, height=height, width=width, output_type="np").images
            
    def test_openclip(self):
        model, _, transform = open_clip.create_model_and_transforms(self.OPENCLIP_MODEL[0], pretrained=self.OPENCLIP_MODEL[1])
        tomeov.patch_openclip(model, 8)
        dummy_image = np.random.rand(224, 224, 3) * 255
        dummy_image = Image.fromarray(dummy_image.astype("uint8"))
        dummy_image = transform(dummy_image).unsqueeze(0)
        
        ov_model = convert_model(
            model.visual,
            example_input=dummy_image
        )
        compiled_model = ov.compile_model(ov_model)
        self.assertTrue(compiled_model)
            
    def test_timm(self):
        model = timm.create_model(self.TIMM_MODEL, pretrained=False)

        tomeov.patch_timm(model, 4) # 8 - number of tokens merged in each MHSA from top down
        
        dummy_image = torch.rand(1, 3, 224, 224)
        
        with tempfile.TemporaryDirectory(suffix = ".onnx") as tmpdirname:
            model_file = os.path.join(tmpdirname, "model.onnx")
            torch.onnx.export(
                model,
                dummy_image,
                model_file,
                opset_version=14,
                input_names=["image"],
                output_names=["output"], 
                dynamic_axes={ 
                    "image": {0: "batch"},
                    "output": {0: "batch"},
                },
                dynamo=False, # This keeps using the classic ONNX exporter (works in PyTorch 1.x – 2.5+).
            )
            compiled_model = ov.compile_model(model_file)
            self.assertTrue(compiled_model)
        