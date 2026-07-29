<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Awesome Robotics with Intel OpenVINO

This document curates a use-case oriented catalog of relevant robotics model deployment with OpenVINO on Intel
platforms.

## Table of contents

- [Awesome Robotics with Intel OpenVINO](#awesome-robotics-with-intel-openvino)
  - [Table of contents](#table-of-contents)
  - [Embodied Reasoning \& Vision-Language Understanding](#embodied-reasoning--vision-language-understanding)
  - [Human-Robot Interaction](#human-robot-interaction)
  - [Multi-Sensor Fusion](#multi-sensor-fusion)
  - [Optical Character Recognition (OCR)](#optical-character-recognition-ocr)
  - [Perception](#perception)
  - [Speech \& Language Interfaces](#speech--language-interfaces)
  - [Vision-Language-Action (VLA) \& Manipulation](#vision-language-action-vla--manipulation)

---

## Embodied Reasoning & Vision-Language Understanding

* **Vision-Language Models**:
  * [BLIP](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/blip-visual-language-processing)
  * [Florence2](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/florence2)
  * [InternVL2](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/internvl2)
  * [LLaVA](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llava-multimodal-chatbot)
  * [LLaVA-NeXT](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llava-next-multimodal-chatbot)
  * [MiniCPM-o](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-o-omnimodal-chatbot)
  * [MiniCPM-V](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot)
  * [Phi-3-Vision](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/phi-3-vision)
  * [Phi-4-Multimodal](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/phi-4-multimodal)
  * [Qwen2-VL](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen2-vl)
  * [Qwen2.5-VL](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen2.5-vl)
  * [Qwen3-VL](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-vl)

## Human-Robot Interaction

* **Gesture, Action & Person Tracking**:
  * [Action Recognition](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/action-recognition-webcam)
  * [Human Pose Estimation](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pose-estimation-webcam)
  * [Person Detection & Re-Identification](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/person-tracking-webcam)
  * [YOLOv8](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/person-counting-webcam)

## Multi-Sensor Fusion

* **Camera-LiDAR BEV Fusion (Autonomous Driving Perception)**:
  * [BEVFusion](../openvino_bevfusion)

## Optical Character Recognition (OCR)

* **Scene Text / Label Reading**:
  * [Handwritten Text Recognition](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/handwritten-ocr)
  * [Horizontal Text Detection & Recognition](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/optical-character-recognition)
  * [PaddleOCR](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddle-ocr-webcam)
  * [PaddleOCR-VL](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl)
  * [Surya](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/surya-line-level-text-detection)
  * [Unlimited-OCR](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/unlimited-ocr)
  * [YOLOv2](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/meter-reader)

## Perception

* **2D Object Detection**:
  * [Vehicle Detection & Attributes Recognition](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vehicle-detection-and-recognition)
  * [YOLOv11](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov11-optimization)
  * [YOLOv26](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization)
  * [YOLOv8](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/object-detection-webcam)
* **3D Object Detection (LiDAR / Point Cloud)**:
  * [PointPillars](../3d/pointPillars)
* **3D Pose & Point Cloud Segmentation**:
  * [Human Pose Estimation 3D](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/3D-pose-estimation-webcam)
  * [PointNet](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/3D-segmentation-point-clouds)
* **6-DoF Object Pose Estimation**:
  * [CDPN](../3d/CDPN)
* **Instance & Semantic Segmentation**:
  * [FastSAM](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/fast-segment-anything)
  * [GroundedSAM](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/grounded-segment-anything)
  * [OneFormer](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/oneformer-segmentation)
  * [RMBG](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/rmbg-background-removal)
  * [Road Segmentation (ADAS)](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-segmentation)
  * [SAM2](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/sam2-image-segmentation)
  * [SAM3](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/sam3)
  * [U^2-Net](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-background-removal)
* **Monocular Depth Estimation**:
  * [Depth Anything](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/depth-anything)
  * [MiDaS](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth)
* **Open-Vocabulary / CLIP-Based Perception**:
  * [CLIP](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/clip-zero-shot-image-classification)
  * [Jina CLIP](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/jina-clip)
  * [MobileCLIP](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/mobileclip-video-search)
  * [SigLIP](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/siglip-zero-shot-image-classification)

## Speech & Language Interfaces

* **ASR (Automatic Speech Recognition / Voice Commands)**:
  * [Distil-Whisper](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/distil-whisper-asr)
  * [FunASR](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/funasr-nano)
  * [MedASR](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/medasr-medical-asr)
  * [MMS](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/mms-massively-multilingual-speech)
  * [Qwen3-ASR](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-asr)
  * [Wav2Vec2](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/speech-recognition-quantization)
  * [Whisper](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-asr-genai)
* **Text-to-Speech (Robot Voice Feedback)**:
  * [Bark](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/bark-text-to-audio)
  * [CosyVoice3](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/cosyvoice3-tts)
  * [FireRedTTS2](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/fireredtts2)
  * [Kokoro](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/kokoro)
  * [OpenVoice](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvoice)
  * [OpenVoice2 & MeloTTS](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvoice2-and-melotts)
  * [Parler TTS](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/parler-tts-text-to-speech)
  * [Qwen3-TTS](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen3-tts)
  * [SpeechT5](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/text-to-speech-genai)
  * [VoxCPM2](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/voxcpm2-tts)

## Vision-Language-Action (VLA) & Manipulation

* **Imitation Learning Policies**:
  * [ACT (ALOHA)](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/aloha-act)
