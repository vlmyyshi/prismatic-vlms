"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

from dataclasses import dataclass
from enum import Enum, unique

from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[str, str]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[str, str]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: str                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        str("llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/xr_core_ai_asl_llm/tree/vlm/data")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        str("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        str("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        str("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        str("llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/xr_core_ai_asl_llm/tree/vlm/data")


@dataclass
class LLaVa_LVIS4V_LRV_POINT_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv-point"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_point.json"),
        str("llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/xr_core_ai_asl_llm/tree/vlm/data")


@dataclass
class LLaVa_LVIS4V_LRV_POINT_20_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv-point-20"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_point_20.json"),
        str("llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/xr_core_ai_asl_llm/tree/vlm/data")


@dataclass
class LLaVa_LVIS4V_LRV_POINT_40_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv-point-40"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_point_40.json"),
        str("llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/xr_core_ai_asl_llm/tree/vlm/data")


@dataclass
class LLaVa_LVIS4V_LRV_POINT_60_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv-point-60"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_point_60.json"),
        str("llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/xr_core_ai_asl_llm/tree/vlm/data")


@dataclass
class LLaVa_LVIS4V_LRV_POINT_80_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv-point-80"

    align_stage_components: Tuple[str, str] = (
        str("download/llava-laion-cc-sbu-558k/chat.json"),
        str("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_point_80.json"),
        str("llava-v1.5-instruct/"),
    )
    dataset_root_dir: str = str("/mnt/xr_core_ai_asl_llm/tree/vlm/data")


@dataclass
class PixMoPoints_Config(DatasetConfig):
    dataset_id: str = "pixmo-points"

    align_stage_components: Tuple[str, str] = (
        str("points-pointing/"),
        str("points-counting/"),
    )
    finetune_stage_components: Tuple[str, str] = (
        str("points-pointing/"),
        str("points-counting/"),
    )
    dataset_root_dir: str = str(
        "/mnt/xr_core_ai_asl_llm/tree/vlm/data/PixMo/torch_datasets/pixmo_datasets"
    )


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    PIXMO_POINTS = PixMoPoints_Config

    LLAVA_LVIS4V_LRV_POINT = LLaVa_LVIS4V_LRV_POINT_Config

    LLAVA_LVIS4V_LRV_POINT_20 = LLaVa_LVIS4V_LRV_POINT_20_Config

    LLAVA_LVIS4V_LRV_POINT_40 = LLaVa_LVIS4V_LRV_POINT_40_Config

    LLAVA_LVIS4V_LRV_POINT_60 = LLaVa_LVIS4V_LRV_POINT_60_Config

    LLAVA_LVIS4V_LRV_POINT_80 = LLaVa_LVIS4V_LRV_POINT_80_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
