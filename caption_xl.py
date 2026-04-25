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
    # Skipping omitted long list of specific images for brevity in this function
    # ... 
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
    
    # --- OFFLINE KAGGLE PATH ---
    LLAVA_MODEL_PATH = "/kaggle/input/datasets/debanikkk/llava-v1-6-mistral-7b-hf"
    
    print("[setup] Caption generation started")
    print(
        f"[setup] image_path={args.image_path} | json_path={args.json_path} | "
        f"output_dir={args.output_dir if args.output_dir else args.json_path} | resolution={image_resolution}"
    )

    print(f"[setup] Loading LLaVA processor and model from {LLAVA_MODEL_PATH}...")
    processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_PATH)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_PATH,
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

        # Upgraded structure-focused prompt
        prompt = (
            "[INST] <image>\nProvide a highly detailed, descriptive caption of this image. "
            "Focus strictly on what is in the scene, the exact layout (e.g., 'on the left', 'in the foreground'), "
            "and the lighting/style. Do not use conversational filler, just the description.[/INST]"
        )
        
        # FIX: Explicitly assigning text= and images= arguments
        inputs = processor(
            text=prompt, 
            images=load_im_from_path(image_path_0), 
            return_tensors="pt"
        ).to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=100)
        prompt1 = processor.decode(output[0], skip_special_tokens=True)
        prompt1 = prompt1.split("[/INST]")[-1].strip()
        
        # FIX: Explicitly assigning text= and images= arguments
        inputs = processor(
            text=prompt, 
            images=load_im_from_path(image_path_1), 
            return_tensors="pt"
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
