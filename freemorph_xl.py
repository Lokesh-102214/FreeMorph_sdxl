import argparse
import json
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from torchvision.utils import save_image
from tqdm import tqdm

from aid_attention import (
    OuterConvergedAttnProcessor_SDPA,
    OuterConvergedAttnProcessor_SDPA2,
    OuterInterpolatedAttnProcessor_SDPA,
)
from aid_utils import (
    fourier_filter,
    generate_beta_tensor,
    linear_interpolation,
    load_im_from_path,
    spherical_interpolation,
)


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def build_fourier_thresholds(total_frames: int, min_threshold: int, max_threshold: int):
    if total_frames < 2:
        raise ValueError("total_frames must be >= 2")
    mid = (total_frames - 1) / 2
    thresholds = []
    for idx in range(total_frames):
        distance = abs(idx - mid) / (mid if mid > 0 else 1)
        threshold = min_threshold + (max_threshold - min_threshold) * distance
        thresholds.append(int(round(threshold)))
    return thresholds


def build_add_time_ids(
    image_height: int,
    image_width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # SDXL UNet uses spatial metadata as time IDs: original size, crop coordinates, target size.
    add_time_ids = torch.tensor(
        [image_height, image_width, 0, 0, image_height, image_width],
        device=device,
        dtype=dtype,
    )
    return add_time_ids.unsqueeze(0).repeat(batch_size, 1)


def encode_prompt_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
):
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
        negative_prompt_2="",
    )
    return (
        prompt_embeds.to(device=device, dtype=dtype),
        negative_prompt_embeds.to(device=device, dtype=dtype),
        pooled_prompt_embeds.to(device=device, dtype=dtype),
        negative_pooled_prompt_embeds.to(device=device, dtype=dtype),
    )

@torch.no_grad()
def aid_inversion(
    unet,
    invert_scheduler,
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    added_cond_kwargs_con: dict,
    inversion_inner_loops: int,
):
    warmup_step = int(len(timesteps) * 0.3)
    warmup_step2 = int(len(timesteps) * 0.6)
    iter_latent = latent.clone()
    for i, t in enumerate(timesteps):
        if i > warmup_step and i < warmup_step2:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA())
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA())
        elif i > warmup_step2:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA2(is_fused=False))
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA2(is_fused=False))
        else:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(AttnProcessor2_0())
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(AttnProcessor2_0())
        for _ in range(inversion_inner_loops):
            noise_pred_cond = unet(
                iter_latent,
                t,
                encoder_hidden_states=text_input_con,
                added_cond_kwargs=added_cond_kwargs_con,
            ).sample
            iter_latent = invert_scheduler.step(sample=latent, model_output=noise_pred_cond, timestep=t).prev_sample
        latent = iter_latent.clone()
    return latent


@torch.no_grad()
def aid_forward(
    unet,
    forward_scheduler,
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncond: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    added_cond_kwargs_con: dict,
    added_cond_kwargs_uncond: dict,
    guidance_scale: float = 3,
):
    attn_processor_dict = {
        "original": {
            "self_attn": AttnProcessor2_0(),
            "cross_attn": AttnProcessor2_0(),
        },
    }
    warmup_step1 = int(len(timesteps) * 0.2)
    warmup_step2 = int(len(timesteps) * 0.6)
    for i, t in enumerate(timesteps):
        if i < warmup_step1:
            interpolate_attn_proc = {
                "self_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_self_attn),
                "cross_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_cross_attn),
            }
        elif i < warmup_step2:
            interpolate_attn_proc = {
                "self_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_self_attn),
                "cross_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_cross_attn),
            }
        else:
            interpolate_attn_proc = {
                "self_attn": AttnProcessor2_0(),
                "cross_attn": AttnProcessor2_0(),
            }
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(interpolate_attn_proc["self_attn"])
            if m_name.endswith("attn2"):
                m.set_processor(interpolate_attn_proc["cross_attn"])
        noise_pred_cond = unet(
            latent,
            t,
            encoder_hidden_states=text_input_con,
            added_cond_kwargs=added_cond_kwargs_con,
        ).sample
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(attn_processor_dict["original"]["self_attn"])
            if m_name.endswith("attn2"):
                m.set_processor(attn_processor_dict["original"]["cross_attn"])
        noise_pred_uncond = unet(
            latent,
            t,
            encoder_hidden_states=text_input_uncond,
            added_cond_kwargs=added_cond_kwargs_uncond,
        ).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = forward_scheduler.step(sample=latent, model_output=noise_pred, timestep=t).prev_sample
    return latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, type=str, help="Path to caption json or its folder")
    parser.add_argument(
        "--model_name",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL checkpoint from Hugging Face",
    )
    parser.add_argument("--image_height", type=int, default=1024, help="Input/output image height")
    parser.add_argument("--image_width", type=int, default=1024, help="Input/output image width")
    parser.add_argument(
        "--num_intermediate_morphs",
        type=int,
        default=8,
        help="Number of generated frames between the two endpoints",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--edit_strength", type=float, default=0.8)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./eval_results/freemorph")
    parser.add_argument("--beta_alpha", type=float, default=20.0)
    parser.add_argument("--beta_beta", type=float, default=20.0)
    parser.add_argument("--fourier_min_threshold", type=int, default=42)
    parser.add_argument("--fourier_max_threshold", type=int, default=48)
    parser.add_argument("--inversion_inner_loops", type=int, default=5)
    args, extras = parser.parse_known_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    dtype_weight = parse_dtype(args.dtype)
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(args.save_dir, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_name, torch_dtype=dtype_weight)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    vae = pipe.vae.to(device=device, dtype=dtype_weight)
    unet = pipe.unet.to(device=device, dtype=dtype_weight)
    forward_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    forward_scheduler.set_timesteps(args.steps, device=device)
    invert_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    invert_scheduler.set_timesteps(args.steps, device=device)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_h = args.image_height // vae_scale_factor
    latent_w = args.image_width // vae_scale_factor
    injection_noise = torch.randn((1, unet.config.in_channels, latent_h, latent_w), device=device, dtype=dtype_weight)

    json_input_path = args.json_path
    if os.path.isdir(json_input_path):
        json_input_path = os.path.join(json_input_path, "caption.json")

    total_frames = args.num_intermediate_morphs + 2
    if total_frames < 3:
        raise ValueError("num_intermediate_morphs must be >= 1")

    with open(json_input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, disable=not accelerator.is_local_main_process)):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_paths = item["image_paths"]
            print(image_paths)
            prompts = item["prompts"]
            exp_id = item["exp_id"]
            latent_x_list = []
            text_input_con_list = []
            text_input_uncond_list = []
            pooled_text_input_con_list = []
            pooled_text_input_uncond_list = []
            original_images = []
            for img_idx in range(2):
                image_path_idx = image_paths[img_idx]
                image = load_im_from_path(image_path_idx, [args.image_height, args.image_width]).to(device, dtype_weight)
                original_images.append(image)

                text_input_con, text_input_uncond, pooled_text_input_con, pooled_text_input_uncond = encode_prompt_sdxl(
                    pipe,
                    prompts[img_idx],
                    device=device,
                    dtype=dtype_weight,
                )
                text_input_con_list.append(text_input_con)
                text_input_uncond_list.append(text_input_uncond)
                pooled_text_input_con_list.append(pooled_text_input_con)
                pooled_text_input_uncond_list.append(pooled_text_input_uncond)
                latent_x = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                latent_x_list.append(latent_x)

            interpolation_size = total_frames
            coef_attn = generate_beta_tensor(interpolation_size, alpha=args.beta_alpha, beta=args.beta_beta)
            latent_x = spherical_interpolation(latent_x_list[0], latent_x_list[1], interpolation_size).squeeze(0)
            invert_timesteps = invert_scheduler.timesteps - 1
            invert_timesteps = invert_timesteps[: int(args.steps * args.edit_strength)]

            text_input_con_all = linear_interpolation(
                text_input_con_list[0], text_input_con_list[1], interpolation_size
            ).squeeze(0)
            text_input_uncond_all = linear_interpolation(
                text_input_uncond_list[0], text_input_uncond_list[1], interpolation_size
            ).squeeze(0)
            pooled_text_input_con_all = linear_interpolation(
                pooled_text_input_con_list[0], pooled_text_input_con_list[1], interpolation_size
            ).squeeze(0)
            pooled_text_input_uncond_all = linear_interpolation(
                pooled_text_input_uncond_list[0], pooled_text_input_uncond_list[1], interpolation_size
            ).squeeze(0)

            add_time_ids = build_add_time_ids(
                image_height=args.image_height,
                image_width=args.image_width,
                batch_size=interpolation_size,
                device=device,
                dtype=dtype_weight,
            )

            reverted_latent = aid_inversion(
                unet=unet,
                invert_scheduler=invert_scheduler,
                latent=latent_x,
                timesteps=invert_timesteps,
                text_input_con=text_input_con_all,
                added_cond_kwargs_con={
                    "text_embeds": pooled_text_input_con_all,
                    "time_ids": add_time_ids,
                },
                inversion_inner_loops=args.inversion_inner_loops,
            )

            thresholds = build_fourier_thresholds(
                total_frames=interpolation_size,
                min_threshold=args.fourier_min_threshold,
                max_threshold=args.fourier_max_threshold,
            )
            new_input_latents_list = []
            for k in range(1, interpolation_size - 1):
                new_input_latent = fourier_filter(
                    x=reverted_latent[k].unsqueeze(0),
                    y=injection_noise.clone(),
                    threshold=int(thresholds[k]),
                )
                new_input_latents_list.append(new_input_latent.clone())
            latent = torch.cat(
                [reverted_latent[0].unsqueeze(0)] + new_input_latents_list + [reverted_latent[-1].unsqueeze(0)]
            )
            forward_timesteps = forward_scheduler.timesteps - 1
            forward_timesteps = forward_timesteps[args.steps - int(args.steps * args.edit_strength) :]
            coef_attn = generate_beta_tensor(interpolation_size, alpha=args.beta_alpha, beta=args.beta_beta)

            morphing_latent = aid_forward(
                unet=unet,
                forward_scheduler=forward_scheduler,
                timesteps=forward_timesteps,
                latent=latent,
                text_input_con=text_input_con_all,
                text_input_uncond=text_input_uncond_all,
                coef_cross_attn=coef_attn,
                coef_self_attn=coef_attn,
                added_cond_kwargs_con={
                    "text_embeds": pooled_text_input_con_all,
                    "time_ids": add_time_ids,
                },
                added_cond_kwargs_uncond={
                    "text_embeds": pooled_text_input_uncond_all,
                    "time_ids": add_time_ids,
                },
                guidance_scale=args.guidance_scale,
            )
            images = []
            for x in morphing_latent:
                x = vae.decode(x.unsqueeze(0) / vae.config.scaling_factor).sample
                x = (x / 2 + 0.5).clamp(0, 1)
                x = x.detach().to(torch.float).cpu()
                images.append(x)
            images[0] = (original_images[0] / 2 + 0.5).detach().to(torch.float).cpu()
            images[-1] = (original_images[1] / 2 + 0.5).detach().to(torch.float).cpu()
            save_image(torch.cat(images), f"{args.save_dir}/{exp_id}.png")
