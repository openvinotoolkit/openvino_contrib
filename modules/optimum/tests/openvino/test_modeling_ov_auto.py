# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from packaging import version

import numpy as np

import transformers
from transformers import AutoTokenizer
import datasets
from datasets import DatasetDict, load_dataset

try:
    from transformers.testing_utils import require_tf, require_torch
except ImportError:
    from transformers.file_utils import is_torch_available, is_tf_available

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
    OVAutoModelForMaskedLM,
    OVAutoModelForQuestionAnswering,
    OVAutoModelWithLMHead,
    OVAutoModelForAudioClassification,
)


class OVBertForQuestionAnsweringTest(unittest.TestCase):
    def check_model(self, model, tok):
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

        self.assertEqual(answer, "the garden")

    @require_torch
    @unittest.skipIf(is_openvino_api_2 and "GITHUB_ACTIONS" in os.environ, "Memory limit exceed")
    def test_from_pt(self):
        tok = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad", from_pt=True
        )
        self.check_model(model, tok)

    @unittest.skipIf(
        version.parse(transformers.__version__) < version.parse("4.0.0"),
        "Too old version of Transformers to test uploaded IR",
    )
    def test_from_ir(self):
        tok = AutoTokenizer.from_pretrained("dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001"
        )
        self.check_model(model, tok)


@require_torch
class GPT2ModelTest(unittest.TestCase):
    def test_model_from_pretrained(self):
        for model_name in ["gpt2"]:
            model = OVAutoModel.from_pretrained(model_name, from_pt=True, use_cache=False)
            self.assertIsNotNone(model)

            input_ids = np.random.randint(0, 255, (1, 6))
            attention_mask = np.random.randint(0, 2, (1, 6))

            expected_shape = (1, 6, 768)
            output = model(input_ids, attention_mask=attention_mask)[0]
            self.assertEqual(output.shape, expected_shape)


@require_torch
class OVAlbertModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head_absolute_embedding(self):
        model = OVAutoModel.from_pretrained("albert-base-v2", from_pt=True)
        input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = (1, 11, 768)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = np.array(
            [[[-0.6513, 1.5035, -0.2766], [-0.6515, 1.5046, -0.2780], [-0.6512, 1.5049, -0.2784]]]
        )

        self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


@require_torch
class OVOPENAIGPTModelLanguageGenerationTest(unittest.TestCase):
    def test_lm_generate_openai_gpt(self):
        model = OVAutoModelWithLMHead.from_pretrained("openai-gpt", from_pt=True)
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


@require_torch
class RobertaModelIntegrationTest(unittest.TestCase):
    def test_inference_masked_lm(self):
        model = OVAutoModelForMaskedLM.from_pretrained("roberta-base", from_pt=True)

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

    def test_inference_no_head(self):
        model = OVAutoModel.from_pretrained("roberta-base", from_pt=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])

        # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        # roberta.eval()
        # expected_slice = roberta.extract_features(input_ids)[:, :3, :3].detach()

        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    def test_inference_batch(self):
        model = OVAutoModel.from_pretrained("roberta-base", from_pt=True)
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


@require_tf
class TFRobertaModelIntegrationTest(unittest.TestCase):
    def test_inference_masked_lm(self):
        model = OVAutoModelForMaskedLM.from_pretrained("roberta-base", from_tf=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = [1, 11, 50265]
        self.assertEqual(list(output.shape), expected_shape)
        # compare the actual values for a slice.
        expected_slice = np.array(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    def test_inference_no_head(self):
        model = OVAutoModel.from_pretrained("roberta-base", from_tf=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


@require_tf
class OVTFDistilBertModelIntegrationTest(unittest.TestCase):
    def test_inference_masked_lm(self):
        model = OVAutoModel.from_pretrained("distilbert-base-uncased", from_tf=True)
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


@require_torch
class OVDistilBertModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head_absolute_embedding(self):
        model = OVAutoModel.from_pretrained("distilbert-base-uncased", from_pt=True)
        model.to(device="CPU")
        model.set_config(config={"CPU_BIND_THREAD": "YES"})

        input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = (1, 11, 768)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = np.array([[[-0.1639, 0.3299, 0.1648], [-0.1746, 0.3289, 0.1710], [-0.1884, 0.3357, 0.1810]]])

        self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


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

    @require_torch
    def test_from_pt(self):
        model = OVAutoModelForAudioClassification.from_pretrained(
            "anton-l/wav2vec2-base-ft-keyword-spotting", from_pt=True
        )
        self.check_model(model)


class OVMBartForConditionalGenerationTest(unittest.TestCase):
    @require_torch
    @unittest.skipIf("GITHUB_ACTIONS" in os.environ, "Memory limit exceed")
    def test_generate(self):
        from optimum.intel.openvino import OVMBartForConditionalGeneration
        from transformers import MBart50TokenizerFast

        model = OVMBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt", from_pt=True
        )
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
        tokenizer.src_lang = "hi_IN"
        encoded_hi = tokenizer(article_hi, return_tensors="pt")
        generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])

        expected_tokens = [
            [
                2,
                250008,
                636,
                21861,
                8,
                96,
                242,
                136840,
                222939,
                1103,
                242,
                379,
                653,
                242,
                53,
                10,
                452,
                8,
                29806,
                128683,
                22,
                51712,
                5,
                2,
            ]
        ]

        self.assertListEqual(generated_tokens.tolist(), expected_tokens)

        decoded_fr = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        self.assertEqual(decoded_fr, "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire en Syria.")
