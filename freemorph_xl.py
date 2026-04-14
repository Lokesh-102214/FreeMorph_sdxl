"""
freemorph_xl.py — FreeMorph ported to Stable Diffusion XL (SDXL)

Key architectural changes vs the original SD 2.1 version:
  • Two CLIP text encoders (ViT-L/14 + ViT-bigG/14) whose hidden states are
    concatenated to produce 2048-dim prompt embeddings.
  • UNet additionally receives added_cond_kwargs = {text_embeds, time_ids}.
  • Native resolution 1024×1024 → latent spatial dim 128×128 (vs 96×96).
  • injection_noise shape dynamically derived from image_resolution.
  • interpolation_size = num_intermediate_morphs + 2  (default: 8+2=10 frames)
  • Fourier thresholds scaled proportionally to latent spatial size.

All previously hard-coded values are now CLI arguments.
Memory / inference optimisations are opt-in boolean flags.

Usage:
    python freemorph_xl.py \
        --json_path captions/caption.json \
        --save_dir  ./results_xl \
        --image_resolution 1024 \
        --num_intermediate_morphs 8 \
        --steps 50 \
        --guidance_scale 7.5 \
        --edit_strength 0.8 \
        --dtype bf16 \
        --enable_xformers \
        --enable_vae_tiling \
        --save_individual
"""

import os
import argparse

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

# SDXL uses two CLIP encoders
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,  # encoder-2 (ViT-bigG); has .text_embeds
    CLIPTokenizer,
)

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

# ─────────────────────────────────────────────────────────────────────────────
# Globals (set in __main__ and used by the two @torch.no_grad() functions)
# ─────────────────────────────────────────────────────────────────────────────
unet             = None
invert_scheduler = None
forward_scheduler = None
cfg              = None          # argparse namespace, injected at startup


# ─────────────────────────────────────────────────────────────────────────────
# SDXL text-encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_prompt_sdxl(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a list of prompts with both SDXL text encoders.

    Returns
    -------
    prompt_embeds   : (B, 77, 2048)  — concat of enc1 + enc2 hidden states
    pooled_embeds   : (B, 1280)      — projected pooled output from enc2
    """
    # ── Encoder 1: CLIP ViT-L/14 (hidden dim 768) ────────────────────────
    tok1 = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids1 = tok1.input_ids.to(device)
    with torch.no_grad():
        # Use penultimate hidden layer ([-2]) per SDXL convention
        h1 = text_encoder(ids1, output_hidden_states=True).hidden_states[-2]  # (B,77,768)

    # ── Encoder 2: CLIP ViT-bigG/14 (hidden dim 1280) ─────────────────────
    tok2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids2 = tok2.input_ids.to(device)
    with torch.no_grad():
        out2   = text_encoder_2(ids2, output_hidden_states=True)
        h2     = out2.hidden_states[-2]   # (B, 77, 1280)
        pooled = out2.text_embeds          # (B, 1280)  — projected pooled output

    # Concatenate along the feature dimension → (B, 77, 2048)
    prompt_embeds = torch.cat([h1, h2], dim=-1).to(dtype)
    pooled_embeds = pooled.to(dtype)
    return prompt_embeds, pooled_embeds


def get_add_time_ids(
    batch_size: int,
    image_h: int,
    image_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build the SDXL time-conditioning vector for every frame in the batch.

    The 6-element vector encodes:
        [orig_height, orig_width, crop_top, crop_left, target_height, target_width]

    When no cropping is performed (center-crop then resize), crop offsets are 0
    and target equals original.

    Returns: (batch_size, 6)
    """
    ids = [image_h, image_w, 0, 0, image_h, image_w]
    return torch.tensor([ids] * batch_size, dtype=dtype, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Fourier threshold schedule (resolution-aware)
# ─────────────────────────────────────────────────────────────────────────────

def build_fourier_thresholds(
    interpolation_size: int,
    latent_h: int,
    latent_w: int,
) -> list[int]:
    """
    Build per-frame Fourier-filter thresholds for the intermediate frames.

    Thresholds taper from a higher value at frames adjacent to the endpoints
    (preserving more low-frequency structure) to a lower value at central frames
    (allowing more injection noise to introduce variation).

    The values are proportionally scaled from the original SD2.1 defaults that
    were tuned for a 96×96 latent (768px image):
        edge  : 48/96 ≈ 0.500
        near  : 44/96 ≈ 0.458
        center: 42/96 ≈ 0.438

    Parameters
    ----------
    interpolation_size : total number of frames including the 2 endpoints
    latent_h, latent_w : spatial dimensions of the latent (image_res // 8)

    Returns
    -------
    List of length `interpolation_size`.  Indices 0 and -1 are 0 (unused;
    those frames keep the inverted latent unchanged).  Indices 1 … -2 carry
    the computed threshold for each intermediate frame.
    """
    lat_dim = min(latent_h, latent_w)
    t_edge   = int(lat_dim * 0.500)   # outermost intermediates
    t_near   = int(lat_dim * 0.458)   # second from each edge
    t_center = int(lat_dim * 0.438)   # all remaining central frames

    thresholds = [0] * interpolation_size  # endpoints: placeholder 0
    for k in range(1, interpolation_size - 1):
        # Symmetric distance from the nearest endpoint among the intermediates
        dist_from_edge = min(k - 1, interpolation_size - 2 - k)
        if dist_from_edge == 0:
            thresholds[k] = t_edge
        elif dist_from_edge == 1:
            thresholds[k] = t_near
        else:
            thresholds[k] = t_center
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# AID Inversion  (SDXL-aware)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def aid_inversion(
    timesteps:        torch.Tensor,
    latent:           torch.Tensor,       # (N, 4, H_lat, W_lat)
    text_input_con:   torch.Tensor,       # (N, 77, 2048)
    text_input_uncon: torch.Tensor,       # (N, 77, 2048)  — unused in inversion but kept for API parity
    pooled_con:       torch.Tensor,       # (N, 1280)
    pooled_uncon:     torch.Tensor,       # (N, 1280)       — unused in inversion
    add_time_ids:     torch.Tensor,       # (N, 6)
    coef_self_attn:   torch.Tensor,
    coef_cross_attn:  torch.Tensor,
    guidance_scale:   float = 7.0,
) -> torch.Tensor:
    """
    AID-guided inversion pass using SDXL UNet.

    Attention-processor schedule (same three-phase structure as the original):
        [0,  30%] : standard AttnProcessor2_0  (warm-up; no attention sharing)
        [30%, 60%]: OuterConvergedAttnProcessor_SDPA  (global convergence)
        [60%, 100%]: OuterConvergedAttnProcessor_SDPA2 (anchored convergence)

    The UNet is called with added_cond_kwargs every step.
    """
    n_steps     = cfg.steps
    warmup_1    = int(n_steps * 0.30)
    warmup_2    = int(n_steps * 0.60)

    added_cond = {"text_embeds": pooled_con, "time_ids": add_time_ids}

    iter_latent = latent.clone()
    for i, t in enumerate(timesteps):
        # ── Attention-processor phase selection ───────────────────────────
        if i <= warmup_1:
            proc = AttnProcessor2_0()
        elif i <= warmup_2:
            proc = OuterConvergedAttnProcessor_SDPA()
        else:
            proc = OuterConvergedAttnProcessor_SDPA2(is_fused=False)

        for m_name, m in unet.named_modules():
            if m_name.endswith(("attn1", "attn2")):
                m.set_processor(proc)

        # ── 5-step iterative inversion refinement ─────────────────────────
        for _ in range(5):
            noise_pred_cond = unet(
                iter_latent, t,
                encoder_hidden_states=text_input_con,
                added_cond_kwargs=added_cond,
            ).sample
            iter_latent = invert_scheduler.step(
                sample=latent,
                model_output=noise_pred_cond,
                timestep=t,
            ).prev_sample
        latent = iter_latent.clone()

    return latent


# ─────────────────────────────────────────────────────────────────────────────
# AID Forward  (SDXL-aware)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def aid_forward(
    timesteps:         torch.Tensor,
    latent:            torch.Tensor,       # (N, 4, H_lat, W_lat)
    text_input_con:    torch.Tensor,       # (N, 77, 2048)
    text_input_uncond: torch.Tensor,       # (N, 77, 2048)
    pooled_con:        torch.Tensor,       # (N, 1280)
    pooled_uncond:     torch.Tensor,       # (N, 1280)
    add_time_ids:      torch.Tensor,       # (N, 6)
    coef_self_attn:    torch.Tensor,
    coef_cross_attn:   torch.Tensor,
    guidance_scale:    float = 3.0,
) -> torch.Tensor:
    """
    AID-guided denoising forward pass using SDXL UNet + CFG.

    Attention-processor schedule:
        [0,   20%]: OuterInterpolated (unfused)  — strong endpoint anchoring
        [20%, 60%]: OuterInterpolated (fused)    — blended with self-attention
        [60%, 100%]: standard AttnProcessor2_0   — free denoising
    """
    base_proc   = AttnProcessor2_0()
    warmup_1    = int(len(timesteps) * 0.20)
    warmup_2    = int(len(timesteps) * 0.60)

    added_con   = {"text_embeds": pooled_con,    "time_ids": add_time_ids}
    added_uncon = {"text_embeds": pooled_uncond,  "time_ids": add_time_ids}

    for i, t in enumerate(timesteps):
        # ── Attention-processor selection for the conditional pass ─────────
        if i < warmup_1:
            interp_self  = OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_self_attn)
            interp_cross = OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_cross_attn)
        elif i < warmup_2:
            interp_self  = OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_self_attn)
            interp_cross = OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_cross_attn)
        else:
            interp_self  = base_proc
            interp_cross = base_proc

        # Conditional pass with interpolated attention
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(interp_self)
            elif m_name.endswith("attn2"):
                m.set_processor(interp_cross)

        noise_pred_cond = unet(
            latent, t,
            encoder_hidden_states=text_input_con,
            added_cond_kwargs=added_con,
        ).sample

        # Reset to standard attention for the unconditional pass
        for m_name, m in unet.named_modules():
            if m_name.endswith(("attn1", "attn2")):
                m.set_processor(base_proc)

        noise_pred_uncond = unet(
            latent, t,
            encoder_hidden_states=text_input_uncond,
            added_cond_kwargs=added_uncon,
        ).sample

        # Classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = forward_scheduler.step(
            sample=latent, model_output=noise_pred, timestep=t
        ).prev_sample

    return latent


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FreeMorph-XL: SDXL-based tuning-free image morphing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ───────────────────────────────────────────────────────────────
    p.add_argument("--json_path", required=True, type=str,
                   help="Path to the JSONL caption file produced by caption_xl.py")
    p.add_argument("--save_dir",  default="./eval_results/freemorph_xl", type=str,
                   help="Directory to write output images")
    p.add_argument("--save_individual", action="store_true",
                   help="Also save each frame as a separate PNG alongside the grid")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model_name", default="stabilityai/stable-diffusion-xl-base-1.0",
                   type=str,
                   help="HuggingFace model ID or local path to an SDXL base model")

    # ── Image / morph geometry ─────────────────────────────────────────────
    p.add_argument("--image_resolution", default=1024, type=int,
                   help="Square I/O resolution in pixels (must be divisible by 8). "
                        "SDXL native: 1024. Lower values save memory at quality cost.")
    p.add_argument("--num_intermediate_morphs", default=8, type=int,
                   help="Number of interpolated frames between the two input images. "
                        "Total output frames = this + 2 (the two endpoints).")

    # ── Diffusion schedule ────────────────────────────────────────────────
    p.add_argument("--steps", default=50, type=int,
                   help="Number of DDIM timesteps used for both inversion and forward")
    p.add_argument("--edit_strength", default=0.8, type=float,
                   help="Fraction [0,1] of timesteps used. "
                        "Higher → more deviation from original images; lower → more faithful.")
    p.add_argument("--guidance_scale", default=7.5, type=float,
                   help="Classifier-free guidance scale for the forward denoising pass")

    # ── Beta distribution for attention schedule ───────────────────────────
    p.add_argument("--beta_alpha", default=20.0, type=float,
                   help="α param of the Beta distribution used for attention coefficients. "
                        "High values (≥10) concentrate weight at the centre of the sequence, "
                        "giving smoother transitions.")
    p.add_argument("--beta_beta",  default=20.0, type=float,
                   help="β param of the Beta distribution (symmetric with --beta_alpha)")

    # ── Dtype ─────────────────────────────────────────────────────────────
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="Weight dtype. bf16 is preferred on Ampere+ GPUs for stability.")

    # ── Misc ──────────────────────────────────────────────────────────────
    p.add_argument("--seed", default=42, type=int,
                   help="Global random seed for reproducibility")

    # ── Memory / inference optimisation flags ─────────────────────────────
    mem = p.add_argument_group("Memory & inference optimisations")
    mem.add_argument("--enable_xformers", action="store_true",
                     help="Enable xformers memory-efficient attention (requires xformers package). "
                          "Significant VRAM saving with negligible quality loss.")
    mem.add_argument("--enable_vae_tiling", action="store_true",
                     help="Tile the VAE decode pass to avoid OOM at 1024px+. "
                          "Slightly slower; no quality impact.")
    mem.add_argument("--enable_vae_slicing", action="store_true",
                     help="Slice the VAE decode batch one image at a time. "
                          "Useful when decoding the full morph strip at once.")
    mem.add_argument("--enable_cpu_offload", action="store_true",
                     help="Offload model components to CPU when not needed "
                          "(model-level; moderate savings, minor slowdown).")
    mem.add_argument("--enable_sequential_offload", action="store_true",
                     help="More aggressive sequential layer-by-layer CPU offload. "
                          "Maximum VRAM saving but noticeably slower. "
                          "Overrides --enable_cpu_offload.")
    mem.add_argument("--compile_unet", action="store_true",
                     help="torch.compile the UNet with mode='reduce-overhead'. "
                          "First batch is slow (compilation); subsequent batches are faster.")
    mem.add_argument("--attention_slicing", action="store_true",
                     help="Slice attention computation head-by-head to reduce peak VRAM. "
                          "Incompatible with xformers; one or the other.")
    mem.add_argument("--channels_last", action="store_true",
                     help="Use torch.channels_last memory format for the UNet. "
                          "Can improve throughput on Ampere+ with fp16/bf16.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()

    # ── Seed & backend flags ───────────────────────────────────────────────
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.deterministic    = False
    torch.set_grad_enabled(False)

    # ── Dtype ─────────────────────────────────────────────────────────────
    _dtype_map   = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype_weight = _dtype_map[cfg.dtype]

    # ── Accelerator ───────────────────────────────────────────────────────
    accelerator = Accelerator()
    device      = accelerator.device
    this_gpu    = accelerator.local_process_index

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ── Derived geometry ──────────────────────────────────────────────────
    assert cfg.image_resolution % 8 == 0, "--image_resolution must be divisible by 8"
    image_resolution  = [cfg.image_resolution, cfg.image_resolution]  # [H, W]
    latent_h          = cfg.image_resolution // 8   # e.g. 1024 → 128
    latent_w          = cfg.image_resolution // 8
    latent_channels   = 4
    # total frames in the morph strip = 2 anchors + N intermediate frames
    interpolation_size = cfg.num_intermediate_morphs + 2

    print(f"[config] model          : {cfg.model_name}")
    print(f"[config] resolution     : {cfg.image_resolution}px → latent {latent_h}×{latent_w}")
    print(f"[config] morph frames   : {interpolation_size} ({cfg.num_intermediate_morphs} intermediate)")
    print(f"[config] steps          : {cfg.steps}  edit_strength={cfg.edit_strength}")
    print(f"[config] guidance_scale : {cfg.guidance_scale}")
    print(f"[config] dtype          : {cfg.dtype}")

    # ── Load SDXL components ──────────────────────────────────────────────
    print("\n[loading] SDXL components…")

    vae = AutoencoderKL.from_pretrained(
        cfg.model_name, subfolder="vae"
    ).to(device, dtype_weight)

    # Text encoders are kept in fp32 for stable gradient flow through embeddings
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.model_name, subfolder="text_encoder"
    ).to(device)

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        cfg.model_name, subfolder="text_encoder_2"
    ).to(device)

    tokenizer   = CLIPTokenizer.from_pretrained(cfg.model_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.model_name, subfolder="tokenizer_2")

    unet = UNet2DConditionModel.from_pretrained(
        cfg.model_name, subfolder="unet"
    ).to(device, dtype_weight)

    forward_scheduler = DDIMScheduler.from_pretrained(cfg.model_name, subfolder="scheduler")
    forward_scheduler.set_timesteps(cfg.steps)

    invert_scheduler = DDIMInverseScheduler.from_pretrained(cfg.model_name, subfolder="scheduler")
    invert_scheduler.set_timesteps(cfg.steps)

    print("[loading] done.")

    # ── Memory / inference optimisations ──────────────────────────────────
    if cfg.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
            print("[opt] xformers memory-efficient attention  ✓")
        except Exception as e:
            print(f"[opt] xformers requested but unavailable: {e}")

    if cfg.channels_last:
        unet = unet.to(memory_format=torch.channels_last)
        print("[opt] channels_last memory format  ✓")

    if cfg.attention_slicing and not cfg.enable_xformers:
        unet.set_attention_slice("auto")
        print("[opt] attention slicing  ✓")
    elif cfg.attention_slicing and cfg.enable_xformers:
        print("[opt] attention_slicing skipped: incompatible with xformers")

    if cfg.enable_vae_tiling:
        vae.enable_tiling()
        print("[opt] VAE tiling  ✓")

    if cfg.enable_vae_slicing:
        vae.enable_slicing()
        print("[opt] VAE slicing  ✓")

    if cfg.enable_sequential_offload:
        # Sequential offload is most aggressive; replaces model-level offload
        unet.enable_sequential_cpu_offload()
        vae.enable_sequential_cpu_offload()
        print("[opt] sequential CPU offload (unet + vae)  ✓")
    elif cfg.enable_cpu_offload:
        unet.enable_model_cpu_offload()
        print("[opt] model-level CPU offload (unet)  ✓")

    if cfg.compile_unet:
        unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
        print("[opt] torch.compile(unet, mode='reduce-overhead')  ✓ (first step will be slow)")

    # ── Static artefacts ──────────────────────────────────────────────────
    # injection_noise: injected into the intermediate latents via Fourier filter
    injection_noise = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        device=device,
        dtype=dtype_weight,
    )

    # Fourier thresholds scaled to current latent resolution
    thresholds = build_fourier_thresholds(interpolation_size, latent_h, latent_w)
    print(f"[config] Fourier thresholds (intermediate frames): {thresholds[1:-1]}")

    # add_time_ids broadcast template (will be tiled per-batch)
    # Actual batch size = interpolation_size, so we build it once here
    add_time_ids_full = get_add_time_ids(
        batch_size=interpolation_size,
        image_h=cfg.image_resolution,
        image_w=cfg.image_resolution,
        device=device,
        dtype=dtype_weight,
    )  # (interpolation_size, 6)

    # ── Main inference loop ───────────────────────────────────────────────
    with open(cfg.json_path) as f:
        for line in tqdm(f, disable=not accelerator.is_local_main_process, desc="pairs"):
            entry       = eval(line)
            image_paths = entry["image_paths"]
            prompts     = entry["prompts"]
            exp_id      = entry["exp_id"]

            print(f"\n[{exp_id}] {image_paths[0]}  ↔  {image_paths[1]}")

            # ── Per-image: encode image + text ────────────────────────────
            latent_x_list      = []
            con_embeds_list    = []
            pooled_con_list    = []
            uncon_embeds_list  = []
            pooled_uncon_list  = []
            original_images    = []

            for img_idx in range(2):
                # Load & VAE-encode image
                image = load_im_from_path(
                    image_paths[img_idx], image_resolution
                ).to(device, dtype_weight)
                original_images.append(image)

                latent = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                latent_x_list.append(latent)  # each: (1, 4, latent_h, latent_w)

                # Conditional text embeddings for this image's prompt
                con_emb, pool_con = encode_prompt_sdxl(
                    prompts=[prompts[img_idx]],
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    device=device,
                    dtype=dtype_weight,
                )  # con_emb: (1,77,2048), pool_con: (1,1280)
                con_embeds_list.append(con_emb)
                pooled_con_list.append(pool_con)

                # Unconditional embeddings (empty string, same for both endpoints)
                uncon_emb, pool_uncon = encode_prompt_sdxl(
                    prompts=[""],
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    device=device,
                    dtype=dtype_weight,
                )  # (1,77,2048), (1,1280)
                uncon_embeds_list.append(uncon_emb)
                pooled_uncon_list.append(pool_uncon)

            # ── Interpolate latents (spherical) ───────────────────────────
            # latent_x_list[i]: (1, 4, latent_h, latent_w)
            # result: (interpolation_size, 4, latent_h, latent_w)
            latent_x = spherical_interpolation(
                latent_x_list[0], latent_x_list[1], interpolation_size
            ).squeeze(0)

            # ── Interpolate text embeddings (linear) ──────────────────────
            # Shape after linear_interpolation: (interpolation_size, 77, 2048)
            text_input_con_all = linear_interpolation(
                con_embeds_list[0], con_embeds_list[1], interpolation_size
            ).squeeze(0)

            # Unconditional: both endpoints are empty-string → embeddings are
            # nearly identical; linear interpolation is harmless but we could
            # also just repeat one of them.
            text_input_uncon_all = linear_interpolation(
                uncon_embeds_list[0], uncon_embeds_list[1], interpolation_size
            ).squeeze(0)

            # Pooled conditional: (interpolation_size, 1280)
            pooled_con_all = linear_interpolation(
                pooled_con_list[0], pooled_con_list[1], interpolation_size
            ).squeeze(0)

            # Pooled unconditional: both empty-string → just repeat endpoint 0
            pooled_uncon_all = pooled_uncon_list[0].expand(interpolation_size, -1).contiguous()

            # ── Beta attention coefficients ────────────────────────────────
            coef_attn = generate_beta_tensor(
                size=interpolation_size,
                alpha=cfg.beta_alpha,
                beta=cfg.beta_beta,
            )  # (interpolation_size,)

            # ── AID Inversion ─────────────────────────────────────────────
            invert_timesteps = invert_scheduler.timesteps - 1
            invert_timesteps = invert_timesteps[: int(cfg.steps * cfg.edit_strength)]

            reverted_latent = aid_inversion(
                timesteps        = invert_timesteps,
                latent           = latent_x,
                text_input_con   = text_input_con_all,
                text_input_uncon = text_input_uncon_all,
                pooled_con       = pooled_con_all,
                pooled_uncon     = pooled_uncon_all,
                add_time_ids     = add_time_ids_full,
                coef_self_attn   = coef_attn,
                coef_cross_attn  = coef_attn,
                guidance_scale   = cfg.guidance_scale,
            )  # (interpolation_size, 4, latent_h, latent_w)

            # ── Fourier filter intermediate frames ────────────────────────
            # Endpoints (idx 0 and -1) keep their inverted latents verbatim.
            # Intermediate frames receive a frequency-domain blend of the
            # inverted latent (low freq) and injection noise (high freq).
            filtered_intermediates = []
            for k in range(1, interpolation_size - 1):
                filtered = fourier_filter(
                    x=reverted_latent[k].unsqueeze(0),
                    y=injection_noise.clone(),
                    threshold=thresholds[k],
                )
                filtered_intermediates.append(filtered)

            latent = torch.cat(
                [reverted_latent[0].unsqueeze(0)]
                + filtered_intermediates
                + [reverted_latent[-1].unsqueeze(0)]
            )  # (interpolation_size, 4, latent_h, latent_w)

            # ── AID Forward (denoising) ────────────────────────────────────
            forward_timesteps = forward_scheduler.timesteps - 1
            forward_timesteps = forward_timesteps[cfg.steps - int(cfg.steps * cfg.edit_strength):]

            # Fresh beta coefficients for the forward pass
            coef_attn = generate_beta_tensor(
                size=interpolation_size,
                alpha=cfg.beta_alpha,
                beta=cfg.beta_beta,
            )

            morphing_latent = aid_forward(
                timesteps         = forward_timesteps,
                latent            = latent,
                text_input_con    = text_input_con_all,
                text_input_uncond = text_input_uncon_all,
                pooled_con        = pooled_con_all,
                pooled_uncond     = pooled_uncon_all,
                add_time_ids      = add_time_ids_full,
                coef_self_attn    = coef_attn,
                coef_cross_attn   = coef_attn,
                guidance_scale    = cfg.guidance_scale,
            )  # (interpolation_size, 4, latent_h, latent_w)

            # ── VAE decode → pixel images ─────────────────────────────────
            images = []
            for x in morphing_latent:
                decoded = vae.decode(x.unsqueeze(0) / vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                images.append(decoded.detach().to(torch.float32).cpu())

            # Replace endpoint decoded images with the original pixel images
            # (bypasses VAE quality loss on the anchor frames)
            images[0]  = (original_images[0] / 2 + 0.5).detach().to(torch.float32).cpu()
            images[-1] = (original_images[1] / 2 + 0.5).detach().to(torch.float32).cpu()

            # ── Save grid (all frames in one horizontal strip) ────────────
            strip = torch.cat(images, dim=0)   # (N, 3, H, W)  nrow handled by save_image
            save_image(strip, f"{cfg.save_dir}/{exp_id}_grid.png", nrow=interpolation_size)
            print(f"  → saved grid  ({interpolation_size} frames)  {cfg.save_dir}/{exp_id}_grid.png")

            # ── Optionally save individual frames ─────────────────────────
            if cfg.save_individual:
                frame_dir = os.path.join(cfg.save_dir, f"{exp_id}_frames")
                os.makedirs(frame_dir, exist_ok=True)
                for fi, img in enumerate(images):
                    if fi == 0:
                        tag = "00_source"
                    elif fi == len(images) - 1:
                        tag = f"{fi:02d}_target"
                    else:
                        tag = f"{fi:02d}_morph"
                    save_image(img, os.path.join(frame_dir, f"{tag}.png"))
                print(f"  → saved {len(images)} individual frames to {frame_dir}/")
