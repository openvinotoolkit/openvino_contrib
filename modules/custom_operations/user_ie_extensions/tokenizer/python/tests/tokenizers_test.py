# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Core
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer


# Left these two methods for convenient transition from legay u8 representation to native string tensors
# TODO: Remove the methods when transition is over
def pack_strings(strings):
    return strings


def unpack_strings(strings):
    return list(strings)


core = Core()

eng_test_strings = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop." "0987654321 - eng, but with d1gits: 123",
    # Qwen tests
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
]
multilingual_test_strings = [
    "Тестовая строка!",
    "Testzeichenfolge?",
    "Tester, la chaîne...",
    "測試字符串",
    "سلسلة الاختبار",
    "מחרוזת בדיקה",
    "Сынақ жолы",
    "رشته تست",
    # Qwen test
    "介绍下清华大学",
    "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
    "糕点商店里原本有三种蛋糕：草莓奶油蛋糕，巧克力椰蓉蛋糕，和红丝绒布朗尼蛋糕。如名字所描述的那样，每种蛋糕都有两种成分：草莓奶油蛋糕包含草莓和奶油两个成分，巧克力椰蓉蛋糕包含巧克力和椰蓉两种成分，红丝绒布朗尼蛋糕包含红丝绒和布朗尼两种成分。在蛋糕制作完成后，往往每一种成分的材料都会有所剩余。为了减少浪费，商店常常会把多出来的成分两两搭配，做成新的小商品卖出去。比如草莓和巧克力可以做成草莓味巧克力酱，布朗尼和椰蓉可以做成布朗尼椰蓉饼干。以此类推可知，如果所有的成分都可以两两组合，那么最终商店能做出哪些小商品出来？",
    "桌子有左中右3个抽屉；张三，李四，王五，赵六都看到桌子上有一袋巧克力。张三让李四和王五出门后，在赵六面前把这袋巧克力放进了右抽屉；王五回来后，张三让赵六出门去找李四，并在王五面前从左抽屉拿出一盒饼干放进中抽屉里；等李四和赵六返回，张三又让王五和赵六出去买酱油，等二人走后，他告诉李四刚才已将一盒饼干放进中抽屉；张三等了很久，发现王五和赵六还没回来，就派李四去寻找，可最后只有王五和李四回来了。王五告诉张三，一开始他们没有找到卖酱油的店，所以只好分头去买，后来赵六走丢了；回来的路上，王五碰上了李四，两人便先赶了回来。于是，张三让两人出门去找赵六；为防再次走丢，张三叮嘱李四和王五要时刻同行，就算酱油买不到，也要找回赵六。结果，李四和王五在外面找到了赵六，发现他已经买了酱油。三人觉得张三从来不出门跑腿，十分气愤，讨论并达成共识，回去见到张三后，不要告诉他买到了酱油的事情，并让王五把酱油藏到自己的背包里。等三人一同回来后，他们按照计划谎称没有买到酱油，并希望张三以后买东西也要一同出门，不能偷懒，张三答应了。当大家最后站在桌子前，四人分别写下自己知道的物品清单和物品所在位置。问，这四人写下的物品和位置信息是否一致，为什么？",
    "折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。首先是创建栅格折痕：这一步有点像我们折千纸鹤的第一步，即通过对称州依次对折，然后按照长和宽两个维度，依次进行多等分的均匀折叠；最终在两个方向上的折痕会交织成一套完整均匀的小方格拼接图案；这些小方格就组成了类似二维坐标系的参考系统，使得我们在该平面上，通过组合临近折痕的方式从二维小方格上折叠出三维的高台或凹陷，以便于接下来的几座制作过程。需要注意的是，在建立栅格折痕的过程中，可能会出现折叠不对成的情况，这种错误所带来的后果可能是很严重的，就像是蝴蝶效应，一开始只是毫厘之差，最后可能就是天壤之别。然后是制作立体基座：在这一步，我们需要基于栅格折痕折出对称的三维高台或凹陷。从对称性分析不难发现，玫瑰花会有四个周对称的三维高台和配套凹陷。所以，我们可以先折出四分之一的凹陷和高台图案，然后以这四分之一的部分作为摸板，再依次折出其余三个部分的重复图案。值得注意的是，高台的布局不仅要考虑长和宽这两个唯独上的规整衬度和对称分布，还需要同时保证高这个维度上的整齐。与第一阶段的注意事项类似，请处理好三个维度上的所有折角，确保它们符合计划中所要求的那种布局，以免出现三维折叠过程中的蝴蝶效应；为此，我们常常会在折叠第一个四分之一图案的过程中，与成品玫瑰花进行反复比较，以便在第一时间排除掉所有可能的错误。最后一个阶段是完成花瓣修饰。在这个阶段，我们往往强调一个重要名词，叫用心折叠。这里的用心已经不是字面上的认真这个意思，而是指通过我们对于大自然中玫瑰花外型的理解，借助自然的曲线去不断修正花瓣的形状，以期逼近现实中的玫瑰花瓣外形。请注意，在这个阶段的最后一步，我们需要通过拉扯已经弯折的四个花瓣，来调整玫瑰花中心的绽放程度。这个过程可能会伴随玫瑰花整体结构的崩塌，所以，一定要控制好调整的力道，以免出现不可逆的后果。最终，经过三个阶段的折叠，我们会得到一支栩栩如生的玫瑰花冠。如果条件允许，我们可以在一根拉直的铁丝上缠绕绿色纸条，并将玫瑰花冠插在铁丝的一段。这样，我们就得到了一支手工玫瑰花。总之，通过创建栅格折痕，制作立体基座，以及完成花瓣修饰，我们从二维的纸面上创作出了一支三维的花朵。这个过程虽然看似简单，但它确实我们人类借助想象力和常见素材而创作出的艺术品。请赏析以上内容的精妙之处。",
]
emoji_test_strings = [
    "😀",
    "😁😁",
    "🤣🤣🤣😁😁😁😁",
    "🫠",  # melting face
    "🤷‍♂️",
    "🤦🏼‍♂️",
]
misc_strings = [
    "",
    b"\x06".decode(),  # control char
    " ",
    " " * 10,
    "\n",
    " \t\n",
]

wordpiece_models = [
    "bert-base-multilingual-cased",
    "bert-large-cased",
    "cointegrated/rubert-tiny2",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "sentence-transformers/all-MiniLM-L6-v2",
    "rajiv003/ernie-finetuned-qqp",  # ernie model with fast tokenizer
    "google/electra-base-discriminator",
    "google/mobilebert-uncased",
    "jhgan/ko-sbert-sts",
    "squeezebert/squeezebert-uncased",
    "prajjwal1/bert-mini",
    "ProsusAI/finbert",
    "rasa/LaBSE",
]
bpe_models = [
    "stabilityai/stablecode-completion-alpha-3b-4k",
    "stabilityai/stablelm-tuned-alpha-7b",
    "databricks/dolly-v2-3b",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-j-6b",
    "roberta-base",
    "sentence-transformers/all-roberta-large-v1",  # standin for setfit
    "facebook/bart-large-mnli",
    "facebook/opt-66b",
    "gpt2",
    "EleutherAI/gpt-neox-20b",
    "ai-forever/rugpt3large_based_on_gpt2",
    "KoboldAI/fairseq-dense-13B",
    "facebook/galactica-120b",
    "EleutherAI/pythia-12b-deduped",
    "microsoft/deberta-base",
    "bigscience/bloom",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "Salesforce/codegen-16B-multi",
    # "google/flan-t5-xxl",  # needs Precompiled/CharsMap
    # "jinmang2/textcnn-ko-dialect-classifier",  # Needs Metaspace Pretokenizer
    # "hyunwoongko/blenderbot-9B",  # hf script to get fast tokenizer doesn't work
]
sentencepiece_models = [
    "codellama/CodeLlama-7b-hf",
    "camembert-base",
    "NousResearch/Llama-2-13b-hf",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "xlnet-base-cased",
    # "THUDM/chatglm-6b",  # hf_tokenizer init error
    "THUDM/chatglm2-6b",  # detokenizer cannot filter special tokens
    "THUDM/chatglm3-6b",
    # "t5-base",  # crashes tests
]
tiktiken_models = [
    "stabilityai/stablelm-2-1_6b",
    "Qwen/Qwen-14B-Chat",
    "Salesforce/xgen-7b-8k-base",
]


def get_tokenizer(hf_tokenizer):
    ov_tokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=False)
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    return hf_tokenizer, compiled_tokenizer


def get_tokenizer_detokenizer(
    hf_tokenizer, streaming_detokenizer=False, skip_special_tokens=False, clean_up_tokenization_spaces=None
):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=True,
        streaming_detokenizer=streaming_detokenizer,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    compiled_detokenizer = core.compile_model(ov_detokenizer)
    return hf_tokenizer, compiled_tokenizer, compiled_detokenizer


def get_hf_tokenizer(request, fast_tokenizer=True, trust_remote_code=False):
    return AutoTokenizer.from_pretrained(request.param, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code)


@pytest.fixture(scope="session", params=wordpiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_wordpiece_tokenizers(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_fast: "Fast" if is_fast else "Slow")
def is_fast_tokenizer(request):
    return request.param


@pytest.fixture(
    scope="session", params=[True, False], ids=lambda do_skip: "skip_tokens" if do_skip else "no_skip_tokens"
)
def do_skip_special_tokens(request):
    return request.param


@pytest.fixture(
    scope="session", params=[True, False], ids=lambda do_clean: "clean_spaces" if do_clean else "no_clean_spaces"
)
def do_clean_up_tokenization_spaces(request):
    return request.param


@pytest.fixture(scope="session", params=sentencepiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_sentencepiece_tokenizers(request, is_fast_tokenizer):
    return get_hf_tokenizer(request, fast_tokenizer=is_fast_tokenizer, trust_remote_code=True)


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_bpe_tokenizers(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session", params=tiktiken_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_tiktoken_tokenizers(request):
    return get_hf_tokenizer(request, trust_remote_code=True)


@pytest.fixture(scope="session")
def wordpiece_tokenizers(hf_wordpiece_tokenizers):
    return get_tokenizer(hf_wordpiece_tokenizers)


@pytest.fixture(scope="session")
def bpe_tokenizers(hf_bpe_tokenizers):
    return get_tokenizer(hf_bpe_tokenizers)


@pytest.fixture(scope="session")
def bpe_tokenizers_detokenizers(hf_bpe_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces):
    return get_tokenizer_detokenizer(
        hf_bpe_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )


@pytest.fixture(scope="session")
def sentencepice_tokenizers(hf_sentencepiece_tokenizers):
    return get_tokenizer(hf_sentencepiece_tokenizers)


@pytest.fixture(scope="session")
def sentencepice_tokenizers_detokenizers(
    hf_sentencepiece_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    # chatglm2 always skips special tokens, chatglam3 always not skip
    if hf_sentencepiece_tokenizers.name_or_path == "THUDM/chatglm2-6b":
        return get_tokenizer_detokenizer(
            hf_sentencepiece_tokenizers,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
        )
    if hf_sentencepiece_tokenizers.name_or_path == "THUDM/chatglm3-6b":
        return get_tokenizer_detokenizer(
            hf_sentencepiece_tokenizers,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
        )

    return get_tokenizer_detokenizer(
        hf_sentencepiece_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )


@pytest.fixture(scope="session")
def tiktoken_tokenizers(hf_tiktoken_tokenizers):
    return get_tokenizer(hf_tiktoken_tokenizers)


@pytest.fixture(scope="session")
def tiktoken_tokenizers_detokenizers(hf_tiktoken_tokenizers, do_skip_special_tokens):
    return get_tokenizer_detokenizer(
        hf_tiktoken_tokenizers, skip_special_tokens=do_skip_special_tokens, clean_up_tokenization_spaces=False
    )


@pytest.fixture(
    scope="session", params=["openlm-research/open_llama_3b_v2"], ids=lambda checkpoint: checkpoint.split("/")[-1]
)
def hf_tokenizers_for_streaming(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session")
def sentencepiece_streaming_tokenizers(hf_tokenizers_for_streaming):
    return get_tokenizer_detokenizer(hf_tokenizers_for_streaming, streaming_detokenizer=True)


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_wordpiece_tokenizers(wordpiece_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = wordpiece_tokenizers
    packed_strings = pack_strings([test_string])

    hf_tokenized = hf_tokenizer([test_string], return_tensors="np")
    ov_tokenized = ov_tokenizer(packed_strings)

    for output_name, hf_result in hf_tokenized.items():
        assert np.all((ov_result := ov_tokenized[output_name]) == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_wordpiece_tokenizers_multiple_strings(wordpiece_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = wordpiece_tokenizers
    packed_strings = pack_strings(test_string)

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np", padding=True)
    ov_tokenized = ov_tokenizer(packed_strings)

    for output_name, hf_result in hf_tokenized.items():
        assert np.all((ov_result := ov_tokenized[output_name]) == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_sentencepiece_model_tokenizer(sentencepice_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = sentencepice_tokenizers

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np")
    ov_tokenized = ov_tokenizer(pack_strings([test_string]))

    for output_name, hf_result in hf_tokenized.items():
        #  chatglm has token_type_ids output that we omit
        if (ov_result := ov_tokenized.get(output_name)) is not None:
            assert np.all(ov_result == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_sentencepiece_model_detokenizer(
    sentencepice_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_tokenizer, _, ov_detokenizer = sentencepice_tokenizers_detokenizers

    token_ids = hf_tokenizer(test_string, return_tensors="np").input_ids
    hf_output = hf_tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )
    ov_output = unpack_strings(ov_detokenizer(token_ids.astype("int32"))["string_output"])

    assert ov_output == hf_output


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_bpe_tokenizers_outputs(bpe_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = bpe_tokenizers
    packed_strings = pack_strings([test_string])

    hf_tokenized = hf_tokenizer([test_string], return_tensors="np")
    ov_tokenized = ov_tokenizer(packed_strings)

    for output_name, hf_result in hf_tokenized.items():
        # galactica tokenizer has 3 output, but model has 2 inputs
        if (ov_result := ov_tokenized.get(output_name)) is not None:
            assert np.all(ov_result == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_bpe_detokenizer(
    bpe_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_tokenizer, _, ov_detokenizer = bpe_tokenizers_detokenizers

    token_ids = hf_tokenizer(test_string, return_tensors="np").input_ids
    hf_output = hf_tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )
    ov_output = unpack_strings(ov_detokenizer(token_ids.astype("int32"))["string_output"])

    assert ov_output == hf_output


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_tokenizers(tiktoken_tokenizers, test_string):
    hf_tokenizer, ov_tokenizer = tiktoken_tokenizers

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np")
    ov_tokenized = ov_tokenizer(pack_strings([test_string]))

    for output_name, hf_result in hf_tokenized.items():
        if (ov_result := ov_tokenized.get(output_name)) is not None:
            assert np.all(ov_result == hf_result), f"{hf_result}\n{ov_result}"


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_detokenizer(tiktoken_tokenizers_detokenizers, test_string, do_skip_special_tokens):
    hf_tokenizer, _, ov_detokenizer = tiktoken_tokenizers_detokenizers

    token_ids = hf_tokenizer(test_string, return_tensors="np").input_ids
    hf_output = hf_tokenizer.batch_decode(token_ids, skip_special_tokens=do_skip_special_tokens)
    ov_output = unpack_strings(ov_detokenizer(token_ids.astype("int32"))["string_output"])

    assert ov_output == hf_output


def test_streaming_detokenizer(sentencepiece_streaming_tokenizers):
    hf_tokenizer, _, ov_detokenizer = sentencepiece_streaming_tokenizers

    test_string = "this is a test string"
    tokenized_string = hf_tokenizer(test_string).input_ids
    hf_detokenized = hf_tokenizer.decode(tokenized_string)

    detokenized_stream = ""
    for token in tokenized_string:
        ov_output = unpack_strings(ov_detokenizer(np.atleast_2d(token))["string_output"])[0]
        detokenized_stream += ov_output

    assert detokenized_stream == hf_detokenized


def test_detokenizer_results_align_with_hf_on_multitoken_symbols_for_streaming():
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat", trust_remote_code=True)
    _, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    ov_detokenizer = core.compile_model(ov_detokenizer)

    test_string = "🤷‍♂️"  # tokenized into 5 tokens
    tokenized_string = hf_tokenizer(test_string).input_ids

    detokenized_stream = ""
    hf_detokenized_stream = ""
    for token in tokenized_string:
        ov_output = unpack_strings(ov_detokenizer(np.atleast_2d(token))["string_output"])[0]
        detokenized_stream += ov_output

        hf_output = hf_tokenizer.decode(token)
        hf_detokenized_stream += hf_output

    assert detokenized_stream == hf_detokenized_stream
