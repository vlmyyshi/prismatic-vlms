"""
scripts/additional-datasets/lvis_instruct4v.py

Standalone script for pre-processing the LVIS-Instruct4V (language/chat) data (`lvis_instruct4v_220k.json`). This
dataset is curated from LVIS images (subset of COCO yet again), but chat data is synthesized from GPT4-Vision.

This script downloads the raw data, merges with the LLaVa v15 data, and performs any other data normalization, saving
the resulting `.json` file(s) to the `data/download/llava-v1.5-instruct/` directory.

Make sure to download the COCO Val 2017 (LVIS) data to `data/download/llava-v1.5-instruct/coco`:
    => cd data/download/llava-v1.5-instruct/coco
    => wget http://images.cocodataset.org/zips/val2017.zip
    => unzip val2017.zip; rm val2017.zip

References: "To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning"
    => Paper: https://arxiv.org/abs/2311.07574
    => Github / Data: https://github.com/X2FD/LVIS-INSTRUCT4V || https://huggingface.co/datasets/X2FD/LVIS-Instruct4V
"""

import json
import os
import random
import re
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image

from prismatic.preprocessing.download import download_with_progress

from tqdm import tqdm

# === Constants ===
DATA_URL = "https://huggingface.co/datasets/X2FD/LVIS-Instruct4V/resolve/main/lvis_instruct4v_220k.json"
DOWNLOAD_DIR = Path("/mnt/xr_core_ai_asl_llm/tree/vlm/data/llava-v1.5-instruct")
RAW_JSON_FILE = DOWNLOAD_DIR / "lvis_instruct4v_220k.json"

# JSON Files for "merged" variant of the dataset (with `llava_v1_5_mix665k.json`)
BASE_DIR = Path("/mnt/xr_core_ai_asl_llm/tree/vlm/data/llava-v1.5-instruct")
BASE_POINTS_FILE = BASE_DIR / "llava_v1_5_point.json"
BASE_LVIS_LRV_JSON_FILE = BASE_DIR / "llava_v1_5_lvis4v_lrv_mix1231k.json"
MERGED_BASE_LVIS_LRV_POINT_JSON_FILE = BASE_DIR / "llava_v1_5_lvis4v_lrv_point.json"
MERGED_BASE_LVIS_LRV_POINT_JSON_20_FILE = (
    BASE_DIR / "llava_v1_5_lvis4v_lrv_point_20.json"
)
MERGED_BASE_LVIS_LRV_POINT_JSON_40_FILE = (
    BASE_DIR / "llava_v1_5_lvis4v_lrv_point_40.json"
)
MERGED_BASE_LVIS_LRV_POINT_JSON_60_FILE = (
    BASE_DIR / "llava_v1_5_lvis4v_lrv_point_60.json"
)
MERGED_BASE_LVIS_LRV_POINT_JSON_80_FILE = (
    BASE_DIR / "llava_v1_5_lvis4v_lrv_point_80.json"
)
TURN_NUM = 2

GENERAL_PROMPTS_V1 = {
    "pointing": [
        "Point to {label}\nPlease say 'This isn't in the image.' if it is not in the image.",
        'Point to all occurrences of "{label}"',
        "Point to any {label} in the image",
        "Point to any {label} in the image.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the image? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        'Find the "{label}".',
        'Find a "{label}".',
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?",
        "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.",
        "Point out every {label} in the image.",
        "Point to the {label} in the image.",
        "Locate each {label} in the image.",
        "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
    ],
    "point_count": [
        "How many {label} are there?",
        "How many {label}?",
        "How many {label}.",
        "how many {label}.",
        "how many {label}?",
        "How many {label} are there in the image?",
        "Tell me how many {label} there are",
        "Tell me how many {label} there are and point to them.",
        "how many {label}",
        "Tell me where each {label} is.",
        "Tell me how many {label} are in the image",
        "count {label}",
        "count every {label}",
        "count each {label}",
        "count {label}.",
        "Count the {label}.",
        "How many {label} do you see?",
        "How many {label} are visible?",
        "Count all the {label}",
        "how mmny {label}?",
        "Count every {label} in the picture.",
        "Count all the {label}",
        "Count each {label}",
        "Point to and count the {label} in the picture.",
        "Point and count {label}",
        "Point to every {label}",
        "Locate the {label} and count them",
        "Locate every {label} and count them",
        "Find all the {label}. How many are there?",
        "Find each {label}. How many are there?",
        "Point at {label} and then tell me the count.",
        "What is the total number of {label} in the image?",
        "In all the picture, how many {label} are there?",
        "Point at the {label} and then count them.",
        "Point to all the visible {label} output the total count.",
        "Point to all the {label} visible and output the total count. \nPlease say 'This isn't in the image.' if it is not in the image.",
        'Point to all occurrences of "{label}" and output the total count.',
        "Show me where the {label} are and output the total count.",
        "Where are the {label}? How many are there?",
        "Generate list of points showing where the {label} are and output the total count.",
        "Object: {label}\nInstruction: Point to the object and output the total count.",
        "find any {label} in the picture and output the total count.",
        "Can you see any {label} in the image? Point to them and output the total count.",
        "Can you point out all {label} in this image? How many are there?",
        "If there are any {label} present, indicate their positions and output the total count.",
        "How many {label} are there in the image? Point to them and output the total count.",
        "How many {label} are there in the image?",
        "Give me the count of {label} in the image.",
        "How many {label} are visible in the image?",
        "How many {label} are there?",
        "In the image, how many {label} are there?",
        "Can you count the number of {label} in the image?",
        "Can you count every {label} in the picture?",
        "Can you see any {label} in the image? How many are there?",
        "Are there any {label} in the image? How many are there?",
        "If you see any {label} in the image, give me the count. Otherwise, say 'This isn't in the image.'",
        "Object: {label}\nInstruction: How many are there?",
    ],
}

wrong_image_ids = [
    693,
    5104,
    9074,
    9594,
    19739,
    51195,
    102767,
    110084,
    116603,
    150348,
    155884,
    166039,
    185888,
]


def apply_keywords(prompt, example, keywords):
    for keyword in keywords:
        res = prompt.split("{" + keyword + "}", maxsplit=2)
        prompt = res[0] + example[keyword] + res[1]
    return prompt


def apply_keyword_prompt(prompts, example, rng, keywords=None):
    if isinstance(prompts, list):
        assert keywords is None
        all_keywords = [sorted(re.findall("{([^{}]+)}", x)) for x in prompts]
        keywords = all_keywords[0]
        assert len(keywords) == len(set(keywords)), f"Repeated keywords in {keywords}"
        assert all(
            keywords == x for x in all_keywords
        ), f"Inconsistent keywords in prompts {all_keywords}"
        assert not any("{" not in word[1:-1] and "}" in word[1:-1] for word in keywords)

        for k in keywords:
            assert (
                k in example
            ), f"Example missing expected field {k}, example={example}"
    new_prompts = []
    for _ in range(TURN_NUM):
        prompt = prompts[rng.randint(0, len(prompts))]
        prompt = apply_keywords(prompt, example, keywords)
        new_prompts.append(prompt)

    return new_prompts


def points_to_text(points, scale, label_text, alt_text):
    if isinstance(scale, (tuple, list)):
        points /= np.array(scale)[None, :]
    else:
        points *= 100 / scale
    points = [[round(x, 1), round(y, 1)] for x, y in points]
    points.sort(key=lambda x: x[0] * 10000 + x[1])
    if len(points) == 1:
        x_str, y_str = points[0]
        return f'<point x="{x_str:0.1f}" y="{y_str:0.1f}" alt="{alt_text}">{label_text}</point>'
    point_text = []
    for ix, (x, y) in enumerate(points, start=1):
        point_text.append(f'x{ix}="{x:0.1f}"')
        point_text.append(f'y{ix}="{y:0.1f}"')
    point_text = " ".join(point_text)
    return f'<points {point_text} alt="{alt_text}">{label_text}</points>'


def format_points(example):
    if "points" not in example:
        return ""
    points = example["points"]
    style = example["style"]
    if "label" in example:
        label = example["label"].lower()
    else:
        label = example["question"]
    if len(points) == 0:
        if style in ["pointing", "point_count"]:
            return "There are none."
        else:
            raise NotImplementedError()
    if "point_scale" in example:
        # Points are already normalized
        point_txt = points_to_text(points, example["point_scale"], label, label)
    else:
        # Points are in pixel coordinate
        h, w = example["image"].shape[:2]
        point_txt = points_to_text(points, [w / 100, h / 100], label, label)

    if style == "point_count":
        return f"Counting the {point_txt} shows a total of {len(points)}."
    else:
        return point_txt


def get_user_prompt(example, rng):
    """Build a list of strings of what a user might type in to the model for the given example,
    and its responses, by applying a prompt template to the fields in `example`
    Uses the `style` field to understand what the task/output style is
    """
    style = example.get("style")
    output = ""
    metadata = None
    if "label" in example:
        prompt = example["label"].lower()
    else:
        prompt = example["label_cased"]
    new_prompts = apply_keyword_prompt(
        GENERAL_PROMPTS_V1[style],
        dict(example, label=prompt),
        rng,
    )
    output = format_points(example)
    return new_prompts, output, metadata


def build_lrv_point_instruct() -> None:
    print("[*] Downloading and Formatting `LRV-points-Instruct` Dataset!")

    # Set Random Seed
    rng = np.random.RandomState(1234)

    ds_pointing = load_from_disk(
        "/home/yyshi/data/molmo/torch_datasets/pixmo_datasets/points-pointing"
    )
    ds_count = load_from_disk(
        "/home/yyshi/data/molmo/torch_datasets/pixmo_datasets/points-counting"
    )
    total_num = (
        len(ds_pointing["train"])
        + len(ds_count["train"])
        + len(ds_pointing["validation"])
        + len(ds_count["validation"])
    )
    chat_json = []
    for idx in tqdm(
        range(total_num), desc="[*] Verifying all VG Images in LRV Instruct"
    ):
        if idx < len(ds_pointing["train"]):
            example = ds_pointing["train"][idx]
        elif idx < len(ds_pointing["train"]) + len(ds_count["train"]):
            example = ds_count["train"][idx - len(ds_pointing["train"])]
        elif idx < len(ds_pointing["train"]) + len(ds_count["train"]) + len(
            ds_pointing["validation"]
        ):
            example = ds_pointing["validation"][
                idx - len(ds_pointing["train"]) - len(ds_count["train"])
            ]
        else:
            example = ds_count["validation"][
                idx
                - len(ds_pointing["train"])
                - len(ds_count["train"])
                - len(ds_pointing["validation"])
            ]

        orig = example["image"]
        image_path = Path(orig)
        # if "/home/yyshi/data/molmo" in orig:
        #     new_root = "/mnt/xr_core_ai_asl_llm/tree/vlm/data/PixMo"
        #     new_path = orig.replace("/home/yyshi/data/molmo", new_root)
        #     image_path = Path(new_path)
        # else:
        #     image_path = Path(orig)
        assert image_path.exists(), f"Missing Image `{image_path}`"
        try:
            Image.open(image_path).convert("RGB")
        except Exception:
            print("Failed to load image", image_path)
            continue
        messages_pointing = []
        messages_point_count = []

        for label, points in zip(example["label"], example["points"]):
            messages_pointing.append(
                dict(
                    label=label,
                    points=np.stack(
                        [[x["x"] for x in points], [x["y"] for x in points]], -1
                    ),
                    point_scale=100,
                    style="pointing",
                )
            )
            messages_point_count.append(
                dict(
                    label=label,
                    points=np.stack(
                        [[x["x"] for x in points], [x["y"] for x in points]], -1
                    ),
                    point_scale=100,
                    style="point_count",
                )
            )

        conversations = []
        for message in messages_pointing + messages_point_count:
            prompts, response, extra_metadata = get_user_prompt(message, rng=rng)
            for prompt in prompts:
                conversations.append(
                    {
                        "from": "human",
                        "value": f"<image>\n{prompt.strip()}",
                    }
                )
                conversations.append({"from": "gpt", "value": response.strip()})

        chat_json.append(
            {
                "id": str(image_path.stem),
                "image": f"../PixMo/torch_datasets/pixmo_images/{image_path.stem}",
                "conversations": conversations,
            }
        )

    with open(BASE_POINTS_FILE, "w") as f:
        json.dump(chat_json, f, indent=2)

    # combine with
    print("[*] Loading LLaVa v1.5 Data!")
    with open(BASE_LVIS_LRV_JSON_FILE, "r") as f:
        llava_v15_data = json.load(f)

    # Combine & Shuffle & Write
    random.seed(7)
    llava_lrv_point_data = llava_v15_data + chat_json

    random.shuffle(llava_lrv_point_data)
    random.shuffle(llava_lrv_point_data)
    random.shuffle(llava_lrv_point_data)
    total_num = len(llava_lrv_point_data)
    print(len(llava_lrv_point_data))

    with open(MERGED_BASE_LVIS_LRV_POINT_JSON_FILE, "w") as f:
        json.dump(llava_lrv_point_data, f, indent=2)

    data_20 = random.sample(llava_lrv_point_data, int(total_num * 0.2))
    with open(MERGED_BASE_LVIS_LRV_POINT_JSON_20_FILE, "w") as f:
        json.dump(data_20, f, indent=2)

    data_40 = random.sample(llava_lrv_point_data, int(total_num * 0.4))
    with open(MERGED_BASE_LVIS_LRV_POINT_JSON_40_FILE, "w") as f:
        json.dump(data_40, f, indent=2)

    data_60 = random.sample(llava_lrv_point_data, int(total_num * 0.6))
    with open(MERGED_BASE_LVIS_LRV_POINT_JSON_60_FILE, "w") as f:
        json.dump(data_60, f, indent=2)

    data_80 = random.sample(llava_lrv_point_data, int(total_num * 0.8))
    with open(MERGED_BASE_LVIS_LRV_POINT_JSON_80_FILE, "w") as f:
        json.dump(data_80, f, indent=2)


if __name__ == "__main__":
    build_lrv_point_instruct()
