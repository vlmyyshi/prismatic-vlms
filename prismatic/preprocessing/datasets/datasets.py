"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type

import datasets
import numpy as np

import torch
from PIL import Image

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform

from prismatic.preprocessing.datasets.dataformatter import DataFormatter
from torch.utils.data import Dataset
from transformers import (
    CodeGenTokenizerFast,
    LlamaTokenizerFast,
    PreTrainedTokenizerBase,
)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        data_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations"]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = (
            Path(self.examples[idx]["image"]),
            self.examples[idx]["conversations"],
        )
        assert (len(conversation) == 2) and (
            "<image>" not in conversation[-1]["value"]
        ), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(
            caption, truncation=True, return_tensors="pt"
        ).input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(
            Image.open(self.image_dir / image_path).convert("RGB")
        )

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum(
                [
                    len(turn["value"].replace("<image>", "").split())
                    for turn in example["conversations"]
                ]
            )
            modality_lengths.append(
                (
                    is_multimodal,
                    (n_image_patches + n_words) if is_multimodal else n_words,
                )
            )
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        data_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = (
            self.prompt_builder_fn(model_family="prismatic"),
            [],
            [],
        )
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()

            # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
            elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                pass

            else:
                raise ValueError(
                    f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!"
                )

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(
                msg, add_special_tokens=turn_idx == 0
            ).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))]
                if (turn_idx % 2) == 0
                else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = (
            input_ids[: self.tokenizer.model_max_length],
            labels[: self.tokenizer.model_max_length],
        )

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            full_path = (self.image_dir / image_path).resolve()
            pixel_values = self.image_transform(Image.open(full_path).convert("RGB"))

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum(
                [len(turn["value"].split()) for turn in example["conversations"]]
            )
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneHFDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        pointing: Path,
        counting: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        data_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"
        self.dataformatter = DataFormatter(
            prompt_templates="uber_model",
            message_format="role",
            system_prompt="demo_or_style",
        )
        self.rng = np.random.RandomState(1234)
        self.mode = "pointing"

        self.pointing_ds = datasets.load_from_disk(pointing, keep_in_memory=False)[
            "train"
        ]
        self.pointing_ds_size = len(self.pointing_ds)
        self.counting_ds = datasets.load_from_disk(counting, keep_in_memory=False)[
            "train"
        ]
        self.counting_ds_size = len(self.counting_ds)
        self.total_size = int(
            float(self.pointing_ds_size + self.counting_ds_size) * data_ratio
        )
        # self.examples = datasets.concatenate_datasets([pointing_ds, counting_ds])

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        if idx >= self.pointing_ds_size:
            idx -= self.pointing_ds_size
            examples = self.counting_ds
        else:
            examples = self.pointing_ds

        ex = examples[idx]
        messages = []
        for label, points in zip(ex["label"], ex["points"]):
            messages.append(
                dict(
                    label=label,
                    points=np.stack(
                        [[x["x"] for x in points], [x["y"] for x in points]], -1
                    ),
                    point_scale=100,
                    style=self.mode,
                )
            )
        input_to_formatter = dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_url=ex["image_url"],
            ),
        )
        formatted, meta_data = self.dataformatter(
            ex=input_to_formatter, is_training=True, for_inference=False, rng=self.rng
        )

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = (
            self.prompt_builder_fn(model_family="prismatic"),
            [],
            [],
        )
        for turn_idx, turn in enumerate(formatted):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()

            # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
            elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                pass

            else:
                raise ValueError(
                    f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!"
                )

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(
                msg, add_special_tokens=turn_idx == 0
            ).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))]
                if (turn_idx % 2) == 0
                else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = (
            input_ids[: self.tokenizer.model_max_length],
            labels[: self.tokenizer.model_max_length],
        )

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in examples[idx]:
            orig = examples[idx]["image"]
            if "/home/yyshi/data/molmo" in orig:
                new_root = "/mnt/xr_core_ai_asl_llm/tree/vlm/data/PixMo"
                new_path = orig.replace("/home/yyshi/data/molmo", new_root)
                image_path = Path(new_path)
            else:
                image_path = Path(orig)

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            try:
                pixel_values = self.image_transform(
                    Image.open(image_path).convert("RGB")
                )
            except Exception:
                print("Failed to load image", image_path)
                mode, size = "RGB", (386, 386)
                fill = (255, 255, 255)
                pixel_values = self.image_transform(Image.new(mode, size, color=fill))

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        rng = np.random.RandomState(1234)
        modality_lengths = []
        for idx in range(0, self.total_size):
            if idx >= self.pointing_ds_size:
                idx -= self.pointing_ds_size
                examples = self.counting_ds
            else:
                examples = self.pointing_ds
            ex = examples[idx]
            is_multimodal = "image" in ex
            messages = []
            for label, points in zip(ex["label"], ex["points"]):
                messages.append(
                    dict(
                        label=label,
                        points=np.stack(
                            [[x["x"] for x in points], [x["y"] for x in points]], -1
                        ),
                        point_scale=100,
                        style=self.mode,
                    )
                )
            input_to_formatter = dict(
                image=ex["image"],
                message_list=messages,
                metadata=dict(
                    image_url=ex["image_url"],
                ),
            )
            formatted, meta_data = self.dataformatter(
                ex=input_to_formatter, is_training=True, for_inference=False, rng=rng
            )
            n_words = sum([len(turn["value"].split()) for turn in formatted])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return self.total_size
