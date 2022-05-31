# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import unittest

import datasets
import numpy as np
import transformers
from datasets import DatasetDict, load_dataset
from packaging import version
from transformers import AutoConfig, AutoTokenizer


try:
    from transformers.testing_utils import require_tf, require_torch
except ImportError:
    from transformers.file_utils import is_tf_available, is_torch_available

    def require_torch(test_case):
        if not is_torch_available():
            return unittest.skip("test requires PyTorch")(test_case)
        else:
            return test_case

    def require_tf(test_case):
        if not is_tf_available():
            return unittest.skip("test requires TensorFlow")(test_case)
        else:
            return test_case


try:
    from openvino.runtime import Core

    Core()
    is_openvino_api_2 = True
except ImportError:
    is_openvino_api_2 = False

from optimum.intel.openvino import (
    OVAutoModel,
    OVAutoModelForAudioClassification,
    OVAutoModelForMaskedLM,
    OVAutoModelForQuestionAnswering,
    OVAutoModelWithLMHead,
    OVMBartForConditionalGeneration,
)

from conftest import start_ovms_with_single_model


class OVBertForQuestionAnsweringTest(unittest.TestCase):
    def check_model(self, model, tok, expected_answer):
        context = """
        Soon her eye fell on a little glass box that
        was lying under the table: she opened it, and
        found in it a very small cake, on which the
        words “EAT ME” were beautifully marked in
        currants. “Well, I’ll eat it,” said Alice, “ and if
        it makes me grow larger, I can reach the key ;
        and if it makes me grow smaller, I can creep
        under the door; so either way I’ll get into the
        garden, and I don’t care which happens !”
        """

        question = "Where Alice should go?"

        # For better OpenVINO efficiency it's recommended to use fixed input shape.
        # So pad input_ids up to specific max_length.
        input_ids = tok.encode(
            question + " " + tok.sep_token + " " + context, return_tensors="np", max_length=128, padding="max_length"
        )

        outputs = model(input_ids)

        start_pos = outputs.start_logits.argmax()
        end_pos = outputs.end_logits.argmax() + 1

        answer_ids = input_ids[0, start_pos:end_pos]
        answer = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(answer_ids))

        self.assertEqual(answer, expected_answer)

    @require_torch
    @unittest.skipIf(is_openvino_api_2 and "GITHUB_ACTIONS" in os.environ, "Memory limit exceed")
    def test_from_pt(self):
        tok = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad", from_pt=True
        )
        self.check_model(model, tok, "the garden")

    @unittest.skipIf(
        version.parse(transformers.__version__) < version.parse("4.0.0"),
        "Too old version of Transformers to test uploaded IR",
    )
    def test_from_ir(self):
        tok = AutoTokenizer.from_pretrained("dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001"
        )
        self.check_model(model, tok, "the garden")

    @require_torch
    def test_tinybert_from_pt(self):
        model_name = "Intel/dynamic_tinybert"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = OVAutoModelForQuestionAnswering.from_pretrained(model_name, from_pt=True)
        self.check_model(model, tok, "if it makes me grow smaller, i can creep under the door")


@require_torch
class MultiFrameworkSameOutputTest(unittest.TestCase):
    def check_same_ouput(self, model_name):
        """
        Sanity check that argmax and argmin output is the same for PyTorch and OpenVINO models
        """
        import torch
        from transformers import AutoModelForMaskedLM, AutoModelForQuestionAnswering
        from utils import generate_random_inputs

        torch.set_grad_enabled(False)

        arch = AutoConfig.from_pretrained(model_name).to_dict()["architectures"][0]
        if arch.endswith("ForQuestionAnswering"):
            ov_model = OVAutoModelForQuestionAnswering.from_pretrained(model_name, from_pt=True)
            pt_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif arch.endswith("ForMaskedLM"):
            ov_model = OVAutoModelForMaskedLM.from_pretrained(model_name, from_pt=True)
            pt_model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            raise NotImplementedError("Only QA and MaskedLM models are currently supported for this test.")
        tok = AutoTokenizer.from_pretrained(model_name)
        input_datas = generate_random_inputs(
            num_items=25, input_names=tok.model_input_names, seq_length=128, return_tensors="pt"
        )

        all_results = []
        for input_data in input_datas:

            ov_result = ov_model(**input_data, return_dict=True)
            pt_result = pt_model(**input_data, return_dict=True)
            ov_values = np.array(next(iter(ov_result.values()))).round(4)
            pt_values = np.array(next(iter(pt_result.values()))).round(4)

            same_result = np.argmax(ov_values) == np.argmax(pt_values) and np.argmin(ov_values) == np.argmin(pt_values)
            all_results.append(same_result)
        all_same = all(all_results)
        self.assertTrue(all_same, f"Different output for PyTorch and OpenVINO on {model_name}: {all_results}")

    def test_same_output_tinybert(self):
        model_name = "Intel/dynamic_tinybert"
        self.check_same_ouput(model_name)

    def test_same_output_bert(self):
        model_name = "csarron/bert-base-uncased-squad-v1"
        self.check_same_ouput(model_name)

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_with_ovms(self):
        try:
            hf_model_name = "dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001"
            ovms_model_name = "bert"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModelForQuestionAnswering)
            config = AutoConfig.from_pretrained(hf_model_name)
            tok = AutoTokenizer.from_pretrained(hf_model_name)
            model = OVAutoModelForQuestionAnswering.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            self.check_model(model, tok)  
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)


@require_torch
class GPT2ModelTest(unittest.TestCase):
    model_kwargs = {"from_pt": True, "use_cache": False}

    def test_model_from_pretrained(self):
        for model_name in ["gpt2"]:
            model = OVAutoModel.from_pretrained(model_name, **self.model_kwargs)
            self.assertIsNotNone(model)

            input_ids = np.random.randint(0, 255, (1, 6))
            attention_mask = np.random.randint(0, 2, (1, 6))

            expected_shape = (1, 6, 768)
            output = model(input_ids, attention_mask=attention_mask)[0]
            self.assertEqual(output.shape, expected_shape)

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_with_ovms(self):
        try:
            for hf_model_name in ["gpt2"]:
                ovms_model_name = hf_model_name
                ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **self.model_kwargs)
                config = AutoConfig.from_pretrained(hf_model_name)
                model = OVAutoModel.from_pretrained(
                    f"localhost:9000/models/{ovms_model_name}",
                    inference_backend="ovms", config=config
                )
                self.assertIsNotNone(model)

                input_ids = np.random.randint(0, 255, (1, 6))
                attention_mask = np.random.randint(0, 2, (1, 6))

                expected_shape = (1, 6, 768)
                output = model(input_ids, attention_mask=attention_mask)[0]
                self.assertEqual(output.shape, expected_shape)

        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)


@require_torch
class OVAlbertModelIntegrationTest(unittest.TestCase):
    model_kwargs = {"from_pt": True}

    def test_inference_no_head_absolute_embedding(self):
        model = OVAutoModel.from_pretrained("albert-base-v2", **self.model_kwargs)
        input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = (1, 11, 768)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = np.array(
            [[[-0.6513, 1.5035, -0.2766], [-0.6515, 1.5046, -0.2780], [-0.6512, 1.5049, -0.2784]]]
        )

        self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_no_head_absolute_embedding_with_ovms(self):
        try:
            hf_model_name = "albert-base-v2"
            ovms_model_name = "albert_base_v2"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModel.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
            attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            output = model(input_ids, attention_mask=attention_mask)[0]
            expected_shape = (1, 11, 768)
            self.assertEqual(output.shape, expected_shape)
            expected_slice = np.array(
                [[[-0.6513, 1.5035, -0.2766], [-0.6515, 1.5046, -0.2780], [-0.6512, 1.5049, -0.2784]]]
            )

            self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)



@require_torch
class OVOPENAIGPTModelLanguageGenerationTest(unittest.TestCase):
    model_kwargs = {"from_pt": True}

    def test_lm_generate_openai_gpt(self):
        model = OVAutoModelWithLMHead.from_pretrained("openai-gpt", **self.model_kwargs)
        input_ids = np.array([[481, 4735, 544]], dtype=np.int64)  # the president is
        expected_output_ids = [
            481,
            4735,
            544,
            246,
            963,
            870,
            762,
            239,
            244,
            40477,
            244,
            249,
            719,
            881,
            487,
            544,
            240,
            244,
            603,
            481,
        ]  # the president is a very good man. " \n " i\'m sure he is, " said the

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @unittest.skip("ovmsclient that does not support torch.Tensor input used in this test")
    def test_lm_generate_openai_gpt_with_ovms(self):
        try:
            hf_model_name = "openai-gpt"
            ovms_model_name = "openai_gpt"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModelWithLMHead, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModelWithLMHead.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            input_ids = np.array([[481, 4735, 544]], dtype=np.int64)  # the president is
            expected_output_ids = [
                481,
                4735,
                544,
                246,
                963,
                870,
                762,
                239,
                244,
                40477,
                244,
                249,
                719,
                881,
                487,
                544,
                240,
                244,
                603,
                481,
            ]  # the president is a very good man. " \n " i\'m sure he is, " said the

            output_ids = model.generate(input_ids, do_sample=False)
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)


@require_torch
class RobertaModelIntegrationTest(unittest.TestCase):
    model_kwargs = {"from_pt": True}

    def test_inference_masked_lm(self):
        model = OVAutoModelForMaskedLM.from_pretrained("roberta-base", **self.model_kwargs)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = (1, 11, 50265)
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = np.array(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )

        # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        # roberta.eval()
        # expected_slice = roberta.model.forward(input_ids)[0][:, :3, :3].detach()
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_masked_lm_with_ovms(self):
        try:
            hf_model_name = "roberta-base"
            ovms_model_name = "roberta_base"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModelForMaskedLM, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModelForMaskedLM.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )

            input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
            output = model(input_ids)[0]
            expected_shape = (1, 11, 50265)
            self.assertEqual(output.shape, expected_shape)
            # compare the actual values for a slice.
            expected_slice = np.array(
                [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
            )

            # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
            # roberta.eval()
            # expected_slice = roberta.model.forward(input_ids)[0][:, :3, :3].detach()
            self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)

    def test_inference_no_head(self):
        model = OVAutoModel.from_pretrained("roberta-base", **self.model_kwargs)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])

        # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        # roberta.eval()
        # expected_slice = roberta.extract_features(input_ids)[:, :3, :3].detach()

        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_no_head_with_ovms(self):
        try:
            hf_model_name = "roberta-base"
            ovms_model_name = "roberta_base"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModel.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
            output = model(input_ids)[0]
            # compare the actual values for a slice.
            expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])

            # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
            # roberta.eval()
            # expected_slice = roberta.extract_features(input_ids)[:, :3, :3].detach()

            self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)

    def test_inference_batch(self):
        model = OVAutoModel.from_pretrained("roberta-base", **self.model_kwargs)
        tok = AutoTokenizer.from_pretrained("roberta-base")

        inputs = ["Good evening.", "here is the sentence I want embeddings for."]
        input_ids = np.concatenate(
            [tok.encode(inp, return_tensors="np", max_length=16, padding="max_length") for inp in inputs]
        )

        output = model(input_ids)[0]

        # compare the actual values for a slice.
        expected_slice = np.array(
            [
                [
                    [-0.09037264, 0.10670696, -0.06938689],
                    [-0.10737953, 0.20106763, 0.04490039],
                    [0.0991028, 0.18117547, 0.0122529],
                ],
                [
                    [-0.09360617, 0.11848789, -0.03424963],
                    [0.1246291, -0.3622079, -0.02089296],
                    [0.36143878, 0.27680993, 0.28920814],
                ],
            ]
        )

        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_batch_with_ovms(self):
        try:
            hf_model_name = "roberta-base"
            ovms_model_name = "roberta_base"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModel.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            tok = AutoTokenizer.from_pretrained("roberta-base")

            inputs = ["Good evening.", "here is the sentence I want embeddings for."]
            input_ids = np.concatenate(
                [tok.encode(inp, return_tensors="np", max_length=16, padding="max_length") for inp in inputs]
            )

            output = model(input_ids)[0]

            # compare the actual values for a slice.
            expected_slice = np.array(
                [
                    [
                        [-0.09037264, 0.10670696, -0.06938689],
                        [-0.10737953, 0.20106763, 0.04490039],
                        [0.0991028, 0.18117547, 0.0122529],
                    ],
                    [
                        [-0.09360617, 0.11848789, -0.03424963],
                        [0.1246291, -0.3622079, -0.02089296],
                        [0.36143878, 0.27680993, 0.28920814],
                    ],
                ]
            )

            self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)


@require_tf
class TFRobertaModelIntegrationTest(unittest.TestCase):
    model_kwargs = {"from_tf": True}

    def test_inference_masked_lm(self):
        model = OVAutoModelForMaskedLM.from_pretrained("roberta-base", **self.model_kwargs)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = [1, 11, 50265]
        self.assertEqual(list(output.shape), expected_shape)
        # compare the actual values for a slice.
        expected_slice = np.array(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_masked_lm_with_ovms(self):
        try:
            hf_model_name = "roberta-base"
            ovms_model_name = "roberta_base"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModelForMaskedLM, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModelForMaskedLM.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )

            input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
            output = model(input_ids)[0]
            expected_shape = [1, 11, 50265]
            self.assertEqual(list(output.shape), expected_shape)
            # compare the actual values for a slice.
            expected_slice = np.array(
                [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
            )
            self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)

    def test_inference_no_head(self):
        model = OVAutoModel.from_pretrained("roberta-base", **self.model_kwargs)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_no_head_with_ovms(self):
        try:
            hf_model_name = "roberta-base"
            ovms_model_name = "roberta_base"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModel.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
            output = model(input_ids)[0]
            # compare the actual values for a slice.
            expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])
            self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)


@require_tf
class OVTFDistilBertModelIntegrationTest(unittest.TestCase):
    model_kwargs = {"from_tf": True}

    def test_inference_masked_lm(self):
        model = OVAutoModel.from_pretrained("distilbert-base-uncased", **self.model_kwargs)
        input_ids = np.array([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        expected_shape = (1, 6, 768)
        self.assertEqual(output.shape, expected_shape)

        expected_slice = np.array(
            [
                [
                    [0.19261885, -0.13732955, 0.4119799],
                    [0.22150156, -0.07422661, 0.39037204],
                    [0.22756018, -0.0896414, 0.3701467],
                ]
            ]
        )
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_masked_lm_with_ovms(self):
        try:
            hf_model_name = "distilbert-base-uncased"
            ovms_model_name = "distilbert_base_uncased"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModel.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
            output = model(input_ids)[0]
            # compare the actual values for a slice.
            expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])
            self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)

@require_torch
class OVDistilBertModelIntegrationTest(unittest.TestCase):
    model_kwargs = {"from_pt": True}

    def test_inference_no_head_absolute_embedding(self):
        model = OVAutoModel.from_pretrained("distilbert-base-uncased", **self.model_kwargs)

        input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = (1, 11, 768)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = np.array([[[-0.1639, 0.3299, 0.1648], [-0.1746, 0.3289, 0.1710], [-0.1884, 0.3357, 0.1810]]])

        self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_inference_no_head_absolute_embedding_with_ovms(self):
        try:
            hf_model_name = "distilbert-base-uncased"
            ovms_model_name = "distilbert_base_uncased"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **self.model_kwargs)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModel.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
            attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            output = model(input_ids, attention_mask=attention_mask)[0]
            expected_shape = (1, 11, 768)
            self.assertEqual(output.shape, expected_shape)
            expected_slice = np.array([[[-0.1639, 0.3299, 0.1648], [-0.1746, 0.3289, 0.1710], [-0.1884, 0.3357, 0.1810]]])

            self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)

@unittest.skipIf(version.parse(transformers.__version__) < version.parse("4.12.0"), "Too old version for Audio models")
class OVAutoModelForAudioClassificationTest(unittest.TestCase):
    def check_model(self, model):
        raw_datasets = DatasetDict()
        raw_datasets["eval"] = load_dataset("superb", "ks", split="validation")
        raw_datasets = raw_datasets.cast_column("audio", datasets.features.Audio(sampling_rate=16000))

        sample = raw_datasets["eval"][0]
        out = model(sample["audio"]["array"].reshape(1, 16000))

        self.assertEqual(np.argmax(out.logits), 11)

    def test_from_ir(self):
        model = OVAutoModelForAudioClassification.from_pretrained("dkurt/wav2vec2-base-ft-keyword-spotting-int8")
        self.check_model(model)

    @unittest.skipIf("TEST_WITH_OVMS" not in os.environ, "OVMS integration tests turned off")
    def test_with_ovms(self):
        try:
            hf_model_name = "dkurt/wav2vec2-base-ft-keyword-spotting-int8"
            ovms_model_name = "wav2vec2"
            ovms_container, tmp_model_dir = start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModelForAudioClassification)
            config = AutoConfig.from_pretrained(hf_model_name)
            model = OVAutoModelForAudioClassification.from_pretrained(
                f"localhost:9000/models/{ovms_model_name}",
                inference_backend="ovms", config=config
            )
            self.check_model(model)
        finally:
            ovms_container.kill()
            shutil.rmtree(tmp_model_dir)

    @require_torch
    def test_from_pt(self):
        model = OVAutoModelForAudioClassification.from_pretrained(
            "anton-l/wav2vec2-base-ft-keyword-spotting", from_pt=True
        )
        self.check_model(model)


@require_torch
@unittest.skipIf("GITHUB_ACTIONS" in os.environ, "Memory limit exceed")
class OVMBartForConditionalGenerationTest(unittest.TestCase):
    def check_model(self, model, expected_fr):
        from transformers import MBart50TokenizerFast

        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
        tokenizer.src_lang = "hi_IN"
        encoded_hi = tokenizer(article_hi, return_tensors="pt")
        generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])

        decoded_fr = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        self.assertEqual(decoded_fr, expected_fr)

    def test_no_cache(self):
        model = OVMBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt", use_cache=False, from_pt=True
        )
        self.check_model(model, "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire en Syria.")

    def test_with_cache(self):
        model = OVMBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt", from_pt=True
        )
        self.check_model(model, "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire en Syria.")

    def test_from_ir(self):
        model = OVMBartForConditionalGeneration.from_pretrained("dkurt/mbart-large-50-many-to-many-mmt-int8")
        self.check_model(model, "Le chef de l'ONU affirme qu'aucune solution militaire n'existe dans la Syrie.")


class QAModelMetricTest(unittest.TestCase):
    def compare_metrics(self, model_name):
        """
        Compute F1 and Exact Match over small dataset subset. Compare OpenVINO
        and PyTorch result: return True if metrics are the same, False if not.
        """
        from utils import QAModel

        ov_model = QAModel(model_name, "ov", 128)
        pt_model = QAModel(model_name, "pt", 128)

        examples = load_dataset("squad", split="validation")
        examples = examples.select(range(0, 50))
        ov_f1, ov_em = ov_model.compute_metrics(examples)
        pt_f1, pt_em = pt_model.compute_metrics(examples)
        print(f"{model_name}, ov f1: {ov_f1}, pt f1: {pt_f1}")
        print(f"{model_name}, ov em: {ov_em}, pt em: {pt_em}")
        return ov_f1 == pt_f1 and ov_em == pt_em

    @require_torch
    @unittest.skipIf(
        version.parse(transformers.__version__) < version.parse("4.15.0"), "Too old version for QA models"
    )
    def test_bert_metrics(self):
        model_name = "csarron/bert-base-uncased-squad-v1"
        self.assertTrue(self.compare_metrics(model_name))
