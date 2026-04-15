import argparse
import json
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

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


def encode_prompt_sdxl(
    prompts: list[str],
    negative_prompts: list[str],
    device,
    dtype,
):
    """
    Returns (prompt_embeds, pooled_prompt_embeds,
             neg_prompt_embeds, neg_pooled_prompt_embeds)
    prompt_embeds shape:  (B, 77, 2048)
    pooled shape:         (B, 1280)
    """

    def _encode(texts, tok1, tok2, enc1, enc2):
        ids1 = tok1(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77,
        ).input_ids.to(device)
        ids2 = tok2(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77,
        ).input_ids.to(device)

        out1 = enc1(ids1, output_hidden_states=True)
        h1 = out1.hidden_states[-2]

        out2 = enc2(ids2, output_hidden_states=True)
        h2 = out2.hidden_states[-2]
        pooled = out2.text_embeds

        embeds = torch.cat([h1, h2], dim=-1).to(dtype)
        pooled = pooled.to(dtype)
        return embeds, pooled

    p_embeds, p_pooled = _encode(prompts, tokenizer, tokenizer_2, text_encoder, text_encoder_2)
    n_embeds, n_pooled = _encode(negative_prompts, tokenizer, tokenizer_2, text_encoder, text_encoder_2)
    return p_embeds, p_pooled, n_embeds, n_pooled


def make_time_ids(batch_size, resolution, device, dtype):
    """SDXL requires original_size, crop_coords, target_size as conditioning."""
    h, w = resolution
    ids = torch.tensor([h, w, 0, 0, h, w], dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1)
    return ids

@torch.no_grad()
def aid_inversion(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncon: torch.Tensor,
    pooled_con: torch.Tensor,
    pooled_uncon: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale=7,
    resolution=(1024, 1024),
):
    batch_size = latent.shape[0]
    warmup_step = int(steps * 0.3)
    warmup_step2 = int(steps * 0.6)
    iter_latent = latent.clone()
    print(
        f"[inversion] start | steps={len(timesteps)} | latent_batch={batch_size} | "
        f"resolution={resolution[0]}x{resolution[1]}"
    )
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

        if i in {0, warmup_step, warmup_step2, len(timesteps) - 1}:
            print(f"[inversion] step {i + 1}/{len(timesteps)} | t={int(t)}")

        time_ids = make_time_ids(batch_size, resolution, latent.device, latent.dtype)
        added_cond_kwargs_con = {
            "text_embeds": pooled_con,
            "time_ids": time_ids,
        }

        for _ in range(5):
            noise_pred_cond = unet(
                iter_latent,
                t,
                encoder_hidden_states=text_input_con,
                added_cond_kwargs=added_cond_kwargs_con,
            ).sample
            iter_latent = invert_scheduler.step(sample=latent, model_output=noise_pred_cond, timestep=t).prev_sample
        latent = iter_latent.clone()
    print("[inversion] done")
    return latent


@torch.no_grad()
def aid_forward(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncond: torch.Tensor,
    pooled_con: torch.Tensor,
    pooled_uncond: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale: float = 3,
    resolution=(1024, 1024),
):
    batch_size = latent.shape[0]
    warmup_step1 = int(len(timesteps) * 0.2)
    warmup_step2 = int(len(timesteps) * 0.6)
    print(
        f"[forward] start | steps={len(timesteps)} | latent_batch={batch_size} | "
        f"resolution={resolution[0]}x{resolution[1]}"
    )
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

        if i in {0, warmup_step1, warmup_step2, len(timesteps) - 1}:
            print(f"[forward] step {i + 1}/{len(timesteps)} | t={int(t)}")

        time_ids = make_time_ids(batch_size, resolution, latent.device, latent.dtype)
        added_cond_con = {"text_embeds": pooled_con, "time_ids": time_ids}
        added_cond_uncon = {"text_embeds": pooled_uncond, "time_ids": time_ids}

        noise_pred_cond = unet(
            latent,
            t,
            encoder_hidden_states=text_input_con,
            added_cond_kwargs=added_cond_con,
        ).sample
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(AttnProcessor2_0())
            if m_name.endswith("attn2"):
                m.set_processor(AttnProcessor2_0())
        noise_pred_uncond = unet(
            latent,
            t,
            encoder_hidden_states=text_input_uncond,
            added_cond_kwargs=added_cond_uncon,
        ).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = forward_scheduler.step(sample=latent, model_output=noise_pred, timestep=t).prev_sample
    print("[forward] done")
    return latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', required=True, type=str)
    parser.add_argument(
        '--output_dir',
        default="./eval_results/freemorph",
        type=str,
        help="where to save output grids",
    )
    parser.add_argument(
        '--num_intermediates',
        default=8,
        type=int,
        help="number of intermediate frames (total = num_intermediates + 2)",
    )
    args, extras = parser.parse_known_args()

    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    interpolation_size = args.num_intermediates + 2
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    image_resolution = 1024
    dtype_weight = torch.float16
    steps = 50
    edit_strength = 0.8
    guidance_scale = 7.5

    accelerater = Accelerator()
    this_gpu = accelerater.local_process_index
    device = accelerater.device

    print("[setup] Starting FreeMorph run")
    print(
        f"[setup] config | json_path={args.json_path} | output_dir={save_dir} | "
        f"num_intermediates={args.num_intermediates} | interpolation_size={interpolation_size}"
    )
    print(f"[setup] model={model_name} | resolution={image_resolution} | device={device} | rank={this_gpu}")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device, dtype_weight)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device, dtype_weight)
    forward_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    forward_scheduler.set_timesteps(steps)
    invert_scheduler = DDIMInverseScheduler.from_pretrained(model_name, subfolder="scheduler")
    invert_scheduler.set_timesteps(steps)

    injection_noise = torch.randn((1, 4, 128, 128), device=device, dtype=dtype_weight)
    print("[setup] Models and schedulers loaded")

    with open(args.json_path) as f:
        for idx, line in enumerate(tqdm(f, disable=not accelerater.is_local_main_process)):
            i = json.loads(line)
            image_paths = i["image_paths"]
            prompts = i["prompts"]
            exp_id = i["exp_id"]

            print(f"[sample {idx}] start | exp_id={exp_id}")
            print(f"[sample {idx}] image_0={image_paths[0]}")
            print(f"[sample {idx}] image_1={image_paths[1]}")

            text_input_con_list = []
            text_input_uncond_list = []
            pooled_con_list = []
            pooled_uncond_list = []
            latent_x_list = []
            original_images = []

            for img_idx in range(2):
                image_path_idx = image_paths[img_idx]
                image = load_im_from_path(image_path_idx, [image_resolution, image_resolution]).to(device, dtype_weight)
                original_images.append(image)

                p_emb, p_pool, n_emb, n_pool = encode_prompt_sdxl(
                    prompts=[prompts[img_idx]],
                    negative_prompts=[""],
                    device=device,
                    dtype=dtype_weight,
                )
                text_input_con_list.append(p_emb)
                text_input_uncond_list.append(n_emb)
                pooled_con_list.append(p_pool)
                pooled_uncond_list.append(n_pool)

                latent_x = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                latent_x_list.append(latent_x)

            print(f"[sample {idx}] encoded images and prompts")

            coef_attn = generate_beta_tensor(interpolation_size, alpha=20, beta=20)
            latent_x = spherical_interpolation(latent_x_list[0], latent_x_list[1], interpolation_size).squeeze(0)

            invert_timesteps = invert_scheduler.timesteps - 1
            invert_timesteps = invert_timesteps[: int(steps * edit_strength)]

            text_input_con_all = linear_interpolation(
                text_input_con_list[0], text_input_con_list[1], interpolation_size
            ).squeeze(0)
            text_input_uncond_all = linear_interpolation(
                text_input_uncond_list[0], text_input_uncond_list[1], interpolation_size
            ).squeeze(0)
            pooled_con_all = linear_interpolation(pooled_con_list[0], pooled_con_list[1], interpolation_size).squeeze(0)
            pooled_uncond_all = linear_interpolation(
                pooled_uncond_list[0], pooled_uncond_list[1], interpolation_size
            ).squeeze(0)

            print(f"[sample {idx}] running inversion")
            reverted_latent = aid_inversion(
                latent=latent_x,
                timesteps=invert_timesteps,
                text_input_con=text_input_con_all,
                text_input_uncon=text_input_uncond_all,
                pooled_con=pooled_con_all,
                pooled_uncon=pooled_uncond_all,
                coef_cross_attn=coef_attn,
                coef_self_attn=coef_attn,
                resolution=(image_resolution, image_resolution),
            )

            n_mid = interpolation_size - 2
            base = [64, 58] + [56] * max(n_mid - 4, 0) + [58, 64]
            if len(base) < n_mid:
                base = base + [56] * (n_mid - len(base))
            elif len(base) > n_mid:
                base = base[:n_mid]
            thresholds = [None] + base + [None]

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
            forward_timesteps = forward_timesteps[steps - int(steps * edit_strength) :]

            print(f"[sample {idx}] running forward diffusion")
            morphing_latent = aid_forward(
                forward_timesteps,
                latent=latent,
                text_input_con=text_input_con_all,
                text_input_uncond=text_input_uncond_all,
                pooled_con=pooled_con_all,
                pooled_uncond=pooled_uncond_all,
                coef_cross_attn=coef_attn,
                coef_self_attn=coef_attn,
                guidance_scale=guidance_scale,
                resolution=(image_resolution, image_resolution),
            )

            images = []
            for x in morphing_latent:
                x = vae.decode(x.unsqueeze(0) / vae.config.scaling_factor).sample
                x = (x / 2 + 0.5).clamp(0, 1).detach().to(torch.float).cpu()
                images.append(x)

            images[0] = (original_images[0] / 2 + 0.5).detach().to(torch.float).cpu()
            images[-1] = (original_images[1] / 2 + 0.5).detach().to(torch.float).cpu()
            out_path = f"{save_dir}/{exp_id}.png"
            save_image(torch.cat(images), out_path)
            print(f"[sample {idx}] saved -> {out_path}")

    print("[done] FreeMorph generation completed")
