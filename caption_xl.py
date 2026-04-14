"""
caption_xl.py — Captioning pipeline for FreeMorph-XL

Generates short phrase-based captions for image pairs using LLaVA-NeXT
and writes a JSONL file compatible with freemorph_xl.py.

The captioner itself (LLaVA-NeXT) is resolution-agnostic, but the images
are resized to --image_resolution before captioning so that the visual
context matches what the diffusion model will process.

Usage:
    python caption_xl.py \
        --image_path /data/image_pairs \
        --json_path  /data/captions \
        --image_resolution 1024 \
        --caption_model llava-hf/llava-v1.6-mistral-7b-hf \
        --max_new_tokens 60 \
        --device cuda \
        --seed 42

Output format (one JSON object per line):
    {"exp_id": 0, "image_paths": ["…_0.jpg", "…_1.jpg"], "prompts": ["…", "…"]}
"""

import argparse
import glob
import json
import os

import torch
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def center_crop(im: Image.Image) -> Image.Image:
    """Square-centre-crop: the longest axis is trimmed to match the shorter."""
    width, height = im.size
    min_dim = min(width, height)
    left   = (width  - min_dim) // 2
    top    = (height - min_dim) // 2
    right  = left + min_dim
    bottom = top  + min_dim
    return im.crop((left, top, right, bottom))


def load_image_for_captioning(im_path: str, image_resolution: int) -> Image.Image:
    """
    Open an image, square-centre-crop, then resize to image_resolution × image_resolution.

    LLaVA-NeXT handles non-square inputs internally, but keeping this consistent
    with the diffusion inference step avoids the captioner seeing regions that
    the diffusion model never processes.
    """
    image = Image.open(im_path).convert("RGB")
    image = center_crop(image)
    image = image.resize((image_resolution, image_resolution), Image.LANCZOS)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Caption image pairs for FreeMorph-XL using LLaVA-NeXT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ───────────────────────────────────────────────────────────────
    p.add_argument("--image_path", required=True, type=str,
                   help="Directory containing paired images "
                        "(naming convention: <name>_0.ext / <name>_1.ext)")
    p.add_argument("--json_path",  required=True, type=str,
                   help="Directory where caption.json will be written")

    # ── Captioner model ───────────────────────────────────────────────────
    p.add_argument("--caption_model", default="llava-hf/llava-v1.6-mistral-7b-hf",
                   type=str,
                   help="HuggingFace model ID or local path for the LLaVA-NeXT captioner. "
                        "Larger variants (e.g. llava-v1.6-34b) produce richer captions "
                        "at the cost of higher VRAM.")

    # ── Image resolution ──────────────────────────────────────────────────
    p.add_argument("--image_resolution", default=1024, type=int,
                   help="Resolution to resize images to before captioning. "
                        "Should match --image_resolution in freemorph_xl.py.")

    # ── Caption generation ────────────────────────────────────────────────
    p.add_argument("--max_new_tokens", default=60, type=int,
                   help="Maximum number of new tokens to generate per caption. "
                        "60 gives roughly 5-8 descriptive phrases.")
    p.add_argument("--num_phrases", default=5, type=int,
                   help="Number of descriptive phrases the model is asked to produce. "
                        "Passed into the prompt template.")

    # ── Device & dtype ────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda", type=str,
                   help="Device for the LLaVA model ('cuda', 'cuda:1', 'cpu')")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="Weight dtype for the LLaVA model")

    # ── Reproducibility ───────────────────────────────────────────────────
    p.add_argument("--seed", default=42, type=int,
                   help="Random seed")

    # ── Memory opts for captioner ─────────────────────────────────────────
    p.add_argument("--load_4bit", action="store_true",
                   help="Load the LLaVA model in 4-bit quantisation via bitsandbytes "
                        "(requires bitsandbytes). Halves VRAM at a minor quality cost.")
    p.add_argument("--load_8bit", action="store_true",
                   help="Load the LLaVA model in 8-bit quantisation via bitsandbytes.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Caption one image
# ─────────────────────────────────────────────────────────────────────────────

def caption_image(
    image: Image.Image,
    processor: LlavaNextProcessor,
    model: LlavaNextForConditionalGeneration,
    device: str,
    max_new_tokens: int,
    num_phrases: int,
) -> str:
    """
    Run LLaVA-NeXT on a single PIL image and return the raw caption string.

    The prompt instructs the model to output exactly `num_phrases` comma-separated
    descriptive phrases, which serves as a structured conditioning signal for
    the diffusion model.
    """
    prompt = (
        f"[INST] <image>\n"
        f"Describe the image using {num_phrases} phrases and separate the phrases using commas."
        f"[/INST]"
    )
    # FIX: Use keyword arguments explicitly
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    # Strip the instruction prefix; only keep what comes after [/INST]
    caption = decoded.split("[/INST]")[-1].strip()
    return caption

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # ── Dtype ─────────────────────────────────────────────────────────────
    _dtype_map   = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype_weight = _dtype_map[args.dtype]

    # ── Build quantisation kwargs if requested ─────────────────────────────
    quant_kwargs: dict = {}
    if args.load_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_weight,
            )
            print("[opt] 4-bit quantisation enabled  ✓")
        except ImportError:
            print("[opt] bitsandbytes not found; ignoring --load_4bit")
    elif args.load_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("[opt] 8-bit quantisation enabled  ✓")
        except ImportError:
            print("[opt] bitsandbytes not found; ignoring --load_8bit")

    # ── Load LLaVA-NeXT ───────────────────────────────────────────────────
    print(f"[loading] {args.caption_model} …")
    processor = LlavaNextProcessor.from_pretrained(args.caption_model)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.caption_model,
        torch_dtype=dtype_weight,
        low_cpu_mem_usage=True,
        **quant_kwargs,
    )
    if not (args.load_4bit or args.load_8bit):
        model = model.to(args.device)
    print("[loading] done.")

    # ── Discover image pairs ───────────────────────────────────────────────
    # Convention: every file ending in _0.<ext> is the "source" image;
    # its counterpart is the same filename with _0 → _1.
    all_files   = sorted(glob.glob(os.path.join(args.image_path, "*")))
    source_imgs = [p for p in all_files if "_0." in os.path.basename(p)]

    if not source_imgs:
        raise FileNotFoundError(
            f"No source images found (pattern *_0.*) in: {args.image_path}"
        )
    print(f"[data] found {len(source_imgs)} image pair(s) in {args.image_path}")

    # ── Caption loop ──────────────────────────────────────────────────────
    json_output = []
    for exp_id, image_path_0 in enumerate(tqdm(source_imgs, desc="captioning")):
        image_path_1 = image_path_0.replace("_0.", "_1.")
        if not os.path.exists(image_path_1):
            print(f"  [warn] partner not found, skipping: {image_path_1}")
            continue

        img_0 = load_image_for_captioning(image_path_0, args.image_resolution)
        img_1 = load_image_for_captioning(image_path_1, args.image_resolution)

        prompt_0 = caption_image(
            img_0, processor, model, args.device, args.max_new_tokens, args.num_phrases
        )
        prompt_1 = caption_image(
            img_1, processor, model, args.device, args.max_new_tokens, args.num_phrases
        )

        json_output.append({
            "exp_id":      exp_id,
            "image_paths": [image_path_0, image_path_1],
            "prompts":     [prompt_0, prompt_1],
        })
        print(f"  [{exp_id}] {os.path.basename(image_path_0)}")
        print(f"         prompt_0: {prompt_0[:80]}…")
        print(f"         prompt_1: {prompt_1[:80]}…")

    # ── Write output ──────────────────────────────────────────────────────
    os.makedirs(args.json_path, exist_ok=True)
    out_path = os.path.join(args.json_path, "caption.json")
    with open(out_path, "w") as f:
        for item in json_output:
            f.write(json.dumps(item) + "\n")
    print(f"\n[done] {len(json_output)} pairs written → {out_path}")
