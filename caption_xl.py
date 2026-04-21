import glob
import json
import os
import random
import shutil

import torch
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm, trange
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import argparse


def center_crop(im: Image) -> Image:
    # Get dimensions
    width, height = im.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_from_path(im_path: str) -> Image:
    image = Image.open(im_path).convert("RGB")
    image = center_crop(image)
    image = image.resize((image_resolution, image_resolution), Image.LANCZOS)
    return image


def get_random_image(folder_path):
    x = random.choice(folder_path)
    while x.split("/")[-1] in [
        "n02085936_7574.JPEG",
        "n02105855_16006.JPEG",
        "n02107574_133.JPEG",
        "n02107683_1677.JPEG",
        "n02107683_5973.JPEG",
        "n02108915_5290.JPEG",
        "n02105641_6159.JPEG",
        "n02123159_630.JPEG",
        "n02124075_7446.JPEG",
        "n02108915_2468.JPEG",
        "n02085936_8369.JPEG",
        "n02108915_2789.JPEG",
        "n02124075_2958.JPEG",
        "n02105855_10368.JPEG",
        "n02123394_2784.JPEG",
        "n02107683_3886.JPEG",
        "n02123394_6318.JPEG",
        "n02105855_13529.JPEG",
        "n02124075_4399.JPEG",
        "n02123597_6444.JPEG",
        "n02123597_2370.JPEG",
        "n02124075_7914.JPEG",
        "n02105855_10167.JPEG",
        "n02123597_1843.JPEG",
        "n02105855_13211.JPEG",
        "n02105855_15330.JPEG",
        "n02107683_5243.JPEG",
        "n02123159_8118.JPEG",
        "n02124075_1953.JPEG",
        "n02107683_3428.JPEG",
        "n02124075_14965.JPEG",
        "n02123597_12906.JPEG",
        "n02123597_8698.JPEG",
        "n02123597_27315.JPEG",
        "n02124075_13216.JPEG",
        "n02123394_2482.JPEG",
        "n02124075_6734.JPEG",
        "n02123394_8271.JPEG",
        "n02123394_4520.JPEG",
        "n02124075_7196.JPEG",
        "n02123597_4513.JPEG",
        "n02123597_2219.JPEG",
        "n02123597_14478.JPEG",
        "n02123597_4550.JPEG",
        "n02105855_5964.JPEG",
        "n02123394_6112.JPEG",
        "n02105855_3240.JPEG",
        "n02107683_2539.JPEG",
        "n02108915_10337.JPEG",
        "n02105855_2447.JPEG",
        "n02105855_16191.JPEG",
        "n02108915_4604.JPEG",
        "n02105855_16951.JPEG",
        "n02105855_18241.JPEG",
        "n02105855_17070.JPEG",
        "n02105855_11493.JPEG",
        "n02107574_3775.JPEG",
        "n02107574_282.JPEG",
        "n02108915_6366.JPEG",
        "n02105855_13382.JPEG",
        "n02123597_7798.JPEG",
        "n02107574_4950.JPEG",
        "n02107574_351.JPEG",
        "n02123597_5513.JPEG",
    ]:
        x = random.choice(folder_path)
    return x


def copy_image(source_folder, dest_folder, image_name):
    dest_path = os.path.join(dest_folder, image_name.split("/")[-1])
    shutil.copy2(image_name, dest_path)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path', required=True, type=str, help="path to the image dir")
    parser.add_argument(
        '--json_path', required=True, type=str, help="path to the caption dir")
    parser.add_argument(
        '--output_dir', default=None, type=str,help="Override output directory for caption.json (defaults to json_path)")
    args, extras = parser.parse_known_args()

    set_seed(42)
    image_resolution = 1024
    print("[setup] Caption generation started")
    print(
        f"[setup] image_path={args.image_path} | json_path={args.json_path} | "
        f"output_dir={args.output_dir if args.output_dir else args.json_path} | resolution={image_resolution}"
    )

    print("[setup] Loading LLaVA processor and model...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
        print(
            f"[setup] Set generation pad_token_id={model.generation_config.pad_token_id} "
            f"(eos_token_id)"
        )
    print("[setup] Model loaded on cuda")

    json_output = []
    all_entries = glob.glob(f"{args.image_path}/*")
    all_images = [p for p in all_entries if os.path.isfile(p)]
    print(f"[run] Found {len(all_entries)} entries ({len(all_images)} files)")
    if len(all_entries) != len(all_images):
        print(f"[run] Skipping {len(all_entries) - len(all_images)} non-file entries")

    for i, image_path in enumerate(tqdm(all_images)):
        base_name = os.path.basename(image_path)
        stem, ext = os.path.splitext(base_name)

        if not ext:
            print(f"[skip] No file extension: {image_path}")
            continue
        if stem.endswith("1"):
            continue
        if not stem.endswith("_0"):
            print(f"[skip] Unexpected naming (expected *_0): {base_name}")
            continue

        paired_stem = f"{stem[:-2]}_1"
        image_path_0 = image_path
        image_path_1 = os.path.join(os.path.dirname(image_path_0), f"{paired_stem}{ext}")
        if not os.path.exists(image_path_1):
            print(f"[skip] Missing pair for {image_path_0}: expected {image_path_1}")
            continue

        prompt = (
            "[INST] <image>\nProvide a highly detailed, descriptive caption of this image. "
            "Focus strictly on what is in the scene, the exact layout (e.g., 'on the left', 'in the foreground'), "
            "and the lighting/style. Do not use conversational filler, just the description.[/INST]"
        )
        inputs = processor(
            prompt, load_im_from_path(image_path_0), return_tensors="pt"
        ).to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=100)
        prompt1 = processor.decode(output[0], skip_special_tokens=True)
        prompt1 = prompt1.split("[/INST]")[-1].strip()
        inputs = processor(
            prompt, load_im_from_path(image_path_1), return_tensors="pt"
        ).to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=100)
        prompt2 = processor.decode(output[0], skip_special_tokens=True)
        prompt2 = prompt2.split("[/INST]")[-1].strip()
        json_output.append(
            {
                "exp_id": i,
                "image_paths": [
                    image_path_0,
                    image_path_1,
                ],
                "prompts": [prompt1, prompt2],
            }
        )
        if (len(json_output) % 10) == 0:
            print(f"[run] Processed {len(json_output)} caption pairs")

    out_dir = args.output_dir if args.output_dir else args.json_path
    os.makedirs(out_dir, exist_ok=True)
    print(f"[save] Writing captions to {out_dir}/caption.json")
    with open(f"{out_dir}/caption.json", "w") as f:
        for item in json_output:
            f.write(json.dumps(item) + "\n")
    print(f"[done] Caption generation completed with {len(json_output)} entries")
