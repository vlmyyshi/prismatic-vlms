"""
generate.py

Simple CLI script to interactively test generating from a pretrained VLM; provides a minimal REPL for specify image
URLs, prompts, and language generation parameters.

Run with: python scripts/generate.py --model_path <PATH TO LOCAL MODEL OR HF HUB>
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

os.environ["TORCH_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = (
    "/mnt/xr_core_ai_asl_llm/tree/vla/models/huggingface/hub"
)
os.environ["HF_HOME"] = "/mnt/xr_core_ai_asl_llm/tree/vla/models/huggingface"

import re

import draccus
import requests
import torch
from PIL import Image, ImageDraw

from prismatic import load
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    model_path: Union[str, Path] = (                                    # Path to Pretrained VLM (on disk or HF Hub)
        "prism-dinosiglip+7b"
    )

    image_dir: str = "/home/yyshi/tmp/"
    output_dir: str = "/home/yyshi/tmp/"

    # Default Generation Parameters =>> subscribes to HuggingFace's GenerateMixIn API
    do_sample: bool = True
    temperature: float = 0.4
    max_new_tokens: int = 512
    min_length: int = 1

    # fmt: on


def highlight_points_on_image(
    img: Image.Image,
    points: torch.Tensor,
    output_path: str,
    radius: int = 5,
    color: tuple = (255, 0, 0),
) -> None:
    """
    Opens the original image, draws a small circle at each (x, y) in `points`,
    and writes the new image to output_path.

    Args:
        image_path:  Path to the original RGB image.
        points:      Tensor of shape (N, 2), in pixel coords (float or int).
        output_path: Where to save the image with points overlaid.
        radius:      Radius of each drawn circle.
        color:       RGB color for the circles (default = red).
    """
    draw = ImageDraw.Draw(img)

    # Each point in `points` is (x, y). Draw a filled circle at each location:
    for x, y in points.tolist():
        x0 = (float(x) / 100) * 224 - radius
        y0 = float(y) * 224 / 100 - radius
        x1 = (float(x) / 100) * 224 + radius
        y1 = float(y) * 224 / 100 + radius
        draw.ellipse([(x0, y0), (x1, y1)], fill=color)

    img.save(output_path)


@draccus.wrap()
def generate(cfg: GenerateConfig) -> None:
    overwatch.info(
        f"Initializing Generation Playground with Prismatic Model `{cfg.model_path}`"
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the pretrained VLM --> uses default `load()` function
    vlm = load(cfg.model_path)
    vlm.to(device, dtype=torch.bfloat16)
    os.makedirs(cfg.output_dir, exist_ok=True)

    for image_file in os.listdir(cfg.image_dir):
        print(image_file)
        image_path = os.path.join(cfg.image_dir, image_file)
        lang = image_file.replace("_", " ")
        match = re.search(r"pick\s+(.*?)\s+from", lang)
        obj = match.group(1)
        img = Image.open(image_path).convert("RGB")
        upsampled = img.resize((224, 224), resample=Image.BILINEAR)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"could you give the points for the {obj} in the image",
        )
        prompt_text = prompt_builder.get_prompt()
        generated_text = vlm.generate(
            upsampled,
            prompt_text,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=512,
            min_length=1,
        )
        print(generated_text)
        m = re.search(r'x="([^"]+)"\s+y="([^"]+)"', generated_text)
        if m:
            x_val, y_val = m.group(1), m.group(2)
            highlight_points_on_image(
                upsampled,
                torch.tensor([[float(x_val), float(y_val)]]),
                output_path=os.path.join(cfg.output_dir, image_file),
            )
        else:
            print("No match")


if __name__ == "__main__":
    generate()
