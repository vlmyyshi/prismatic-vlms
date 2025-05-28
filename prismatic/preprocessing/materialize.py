"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from prismatic.conf import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.preprocessing.datasets import (
    AlignDataset,
    FinetuneDataset,
    FinetuneHFDataset,
)
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {
    "align": AlignDataset,
    "finetune": FinetuneDataset,
    "full-finetune": FinetuneDataset,
    "pixmo-finetune": FinetuneHFDataset,
}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        default_image_resolution,
        padding_side=padding_side,
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            Path(dataset_root_dir) / Path(annotation_json),
            Path(dataset_root_dir) / Path(image_dir),
            image_transform,
            tokenizer,
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            Path(dataset_root_dir) / Path(annotation_json),
            Path(dataset_root_dir) / Path(image_dir),
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            Path(dataset_root_dir) / Path(annotation_json),
            Path(dataset_root_dir) / Path(image_dir),
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "pixmo-finetune":
        pointing, counting = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            Path(dataset_root_dir) / Path(pointing),
            Path(dataset_root_dir) / Path(counting),
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
