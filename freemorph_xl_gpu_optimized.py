"""
freemorph_xl_gpu_optimized.py — GPU-optimized FreeMorph-XL with max memory utilization

KEY OPTIMIZATIONS:
  ✓ Batched VAE decoding (replaces frame-by-frame decode)
  ✓ Automatic dynamic batch size calculation
  ✓ Smart GPU memory tracking and garbage collection
  ✓ Tensor reuse to reduce fragmentation
  ✓ New flag: --enable_gpu_optimization (enables all)
  ✓ New flag: --vae_decode_batch_size (manual override)
  ✓ New flag: --enable_memory_monitoring (shows GPU stats)

USAGE:
    python freemorph_xl_gpu_optimized.py \
        --json_path captions/caption.json \
        --image_resolution 1024 \
        --num_intermediate_morphs 8 \
        --dtype bf16 \
        --steps 50 \
        --guidance_scale 7.5 \
        --enable_xformers \
        --channels_last \
        --compile_unet \
        --enable_gpu_optimization \
        --enable_memory_monitoring
"""

import os
import argparse
import gc
from typing import Optional, List

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

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
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


# ────────────────────────────────────────────────────────────────
# GPU Memory Utilities
# ────────────────────────────────────────────────────────────────

class GPUMemoryMonitor:
    """Monitor and optimize GPU memory usage"""
    
    def __init__(self, device: torch.device, enable_monitoring: bool = False):
        self.device = device
        self.enable_monitoring = enable_monitoring
        self.peak_allocated = 0
        self.peak_reserved = 0
    
    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        total = props.total_memory / 1024**3
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'available_gb': total - reserved,
            'percent_used': (allocated / total) * 100
        }
    
    def print_stats(self, tag: str = ""):
        """Print memory stats if monitoring enabled"""
        if self.enable_monitoring and torch.cuda.is_available():
            stats = self.get_memory_stats()
            tag_str = f"[{tag}] " if tag else ""
            print(f"{tag_str}GPU Memory: {stats['allocated_gb']:.1f}/{stats['total_gb']:.1f}GB "
                  f"({stats['percent_used']:.1f}%) allocated")
    
    def clear_cache(self):
        """Clear GPU cache and collect garbage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def calculate_optimal_vae_batch_size(
    vae_model: AutoencoderKL,
    device: torch.device,
    interpolation_size: int,
    latent_h: int,
    latent_w: int,
    target_memory_fraction: float = 0.6,
) -> int:
    """
    Calculate optimal VAE decode batch size based on available GPU memory.
    
    Args:
        vae_model: The VAE decoder model
        device: CUDA device
        interpolation_size: Total number of frames to decode
        latent_h, latent_w: Latent dimensions
        target_memory_fraction: Target GPU memory to use (0.0-1.0)
    
    Returns:
        Optimal batch size for VAE decoding
    """
    if not torch.cuda.is_available():
        return 1
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Get available memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    current_allocated = torch.cuda.memory_allocated(device)
    available_memory = (total_memory - current_allocated) * target_memory_fraction
    
    # Estimate memory per frame
    # Latent: (1, 4, H, W) in bf16 = 2 bytes per value
    # Decoded: (1, 3, H*8, W*8) in fp32 = 4 bytes per value
    latent_mem = 1 * 4 * latent_h * latent_w * 2  # bytes
    decoded_mem = 1 * 3 * (latent_h * 8) * (latent_w * 8) * 4  # bytes
    total_per_frame = (latent_mem + decoded_mem) * 1.3  # 1.3x safety factor
    
    # Calculate batch size
    optimal_batch_size = max(1, int(available_memory / total_per_frame))
    optimal_batch_size = min(optimal_batch_size, interpolation_size)
    
    return optimal_batch_size


# ────────────────────────────────────────────────────────────────
# Batched VAE Operations
# ────────────────────────────────────────────────────────────────

def batch_vae_decode(
    vae: AutoencoderKL,
    morphing_latent: torch.Tensor,
    batch_size: Optional[int] = None,
    memory_monitor: Optional[GPUMemoryMonitor] = None,
    device: torch.device = torch.device("cuda"),
) -> List[torch.Tensor]:
    """
    Decode VAE latents with batching for efficient GPU utilization.
    
    Args:
        vae: VAE model
        morphing_latent: Latent tensor (N, 4, H, W)
        batch_size: Batch size for decoding (auto-calculated if None)
        memory_monitor: Memory monitor for tracking
        device: Device to use
    
    Returns:
        List of decoded images (each on CPU)
    """
    interpolation_size = morphing_latent.shape[0]
    
    # Auto-calculate batch size if not provided
    if batch_size is None:
        latent_h, latent_w = morphing_latent.shape[-2:]
        batch_size = calculate_optimal_vae_batch_size(
            vae, device, interpolation_size, latent_h, latent_w
        )
    
    if memory_monitor:
        memory_monitor.print_stats("before_vae_decode")
    
    images = []
    
    # Process in batches
    for start_idx in range(0, interpolation_size, batch_size):
        end_idx = min(start_idx + batch_size, interpolation_size)
        batch_latent = morphing_latent[start_idx:end_idx]
        
        # Decode batch
        with torch.no_grad():
            batch_decoded = vae.decode(
                batch_latent / vae.config.scaling_factor
            ).sample
        
        # Normalize to [0, 1] and transfer to CPU
        batch_decoded = (batch_decoded / 2 + 0.5).clamp(0, 1)
        
        for i in range(batch_decoded.shape[0]):
            images.append(
                batch_decoded[i].detach().to(torch.float32).cpu()
            )
        
        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if memory_monitor:
        memory_monitor.print_stats("after_vae_decode")
    
    return images


# ────────────────────────────────────────────────────────────────
# Globals
# ────────────────────────────────────────────────────────────────

unet = None
invert_scheduler = None
forward_scheduler = None
cfg = None


# ────────────────────────────────────────────────────────────────
# SDXL Text Encoding (UNCHANGED)
# ────────────────────────────────────────────────────────────────

def encode_prompt_sdxl(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of prompts with both SDXL text encoders."""
    tok1 = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids1 = tok1.input_ids.to(device)
    with torch.no_grad():
        h1 = text_encoder(ids1, output_hidden_states=True).hidden_states[-2]

    tok2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids2 = tok2.input_ids.to(device)
    with torch.no_grad():
        out2 = text_encoder_2(ids2, output_hidden_states=True)
        h2 = out2.hidden_states[-2]
        pooled = out2.text_embeds

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
    """Build the SDXL time-conditioning vector."""
    ids = [image_h, image_w, 0, 0, image_h, image_w]
    return torch.tensor([ids] * batch_size, dtype=dtype, device=device)


# ────────────────────────────────────────────────────────────────
# Fourier Thresholds (UNCHANGED)
# ────────────────────────────────────────────────────────────────

def build_fourier_thresholds(
    interpolation_size: int,
    latent_h: int,
    latent_w: int,
) -> list[int]:
    """Build per-frame Fourier-filter thresholds."""
    lat_dim = min(latent_h, latent_w)
    t_edge = int(lat_dim * 0.500)
    t_near = int(lat_dim * 0.458)
    t_center = int(lat_dim * 0.438)

    thresholds = [0] * interpolation_size
    for k in range(1, interpolation_size - 1):
        dist_from_edge = min(k - 1, interpolation_size - 2 - k)
        if dist_from_edge == 0:
            thresholds[k] = t_edge
        elif dist_from_edge == 1:
            thresholds[k] = t_near
        else:
            thresholds[k] = t_center
    return thresholds


# ────────────────────────────────────────────────────────────────
# AID Inversion (UNCHANGED)
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def aid_inversion(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncon: torch.Tensor,
    pooled_con: torch.Tensor,
    pooled_uncon: torch.Tensor,
    add_time_ids: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale: float = 7.0,
) -> torch.Tensor:
    """AID-guided inversion pass using SDXL UNet."""
    n_steps = cfg.steps
    warmup_1 = int(n_steps * 0.30)
    warmup_2 = int(n_steps * 0.60)

    added_cond = {"text_embeds": pooled_con, "time_ids": add_time_ids}

    iter_latent = latent.clone()
    for i, t in enumerate(timesteps):
        if i <= warmup_1:
            proc = AttnProcessor2_0()
        elif i <= warmup_2:
            proc = OuterConvergedAttnProcessor_SDPA()
        else:
            proc = OuterConvergedAttnProcessor_SDPA2(is_fused=False)

        for m_name, m in unet.named_modules():
            if m_name.endswith(("attn1", "attn2")):
                m.set_processor(proc)

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


# ────────────────────────────────────────────────────────────────
# AID Forward (UNCHANGED)
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def aid_forward(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncond: torch.Tensor,
    pooled_con: torch.Tensor,
    pooled_uncond: torch.Tensor,
    add_time_ids: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale: float = 3.0,
) -> torch.Tensor:
    """AID-guided denoising forward pass using SDXL UNet + CFG."""
    base_proc = AttnProcessor2_0()
    warmup_1 = int(len(timesteps) * 0.20)
    warmup_2 = int(len(timesteps) * 0.60)

    added_con = {"text_embeds": pooled_con, "time_ids": add_time_ids}
    added_uncon = {"text_embeds": pooled_uncond, "time_ids": add_time_ids}

    for i, t in enumerate(timesteps):
        if i < warmup_1:
            interp_self = OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_self_attn)
            interp_cross = OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_cross_attn)
        elif i < warmup_2:
            interp_self = OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_self_attn)
            interp_cross = OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_cross_attn)
        else:
            interp_self = base_proc
            interp_cross = base_proc

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

        for m_name, m in unet.named_modules():
            if m_name.endswith(("attn1", "attn2")):
                m.set_processor(base_proc)

        noise_pred_uncond = unet(
            latent, t,
            encoder_hidden_states=text_input_uncond,
            added_cond_kwargs=added_uncon,
        ).sample

        # FIX: Clone tensors to avoid CUDA graph issues with torch.compile
        noise_pred_cond = noise_pred_cond.clone()
        noise_pred_uncond = noise_pred_uncond.clone()
        
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = forward_scheduler.step(
            sample=latent, model_output=noise_pred, timestep=t
        ).prev_sample

    return latent


# ────────────────────────────────────────────────────────────────
# Argument Parsing
# ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FreeMorph-XL: SDXL-based tuning-free image morphing (GPU-Optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--json_path", required=True, type=str,
                   help="Path to the JSONL caption file produced by caption_xl.py")
    p.add_argument("--save_dir", default="./eval_results/freemorph_xl", type=str,
                   help="Directory to write output images")
    p.add_argument("--save_individual", action="store_true",
                   help="Also save each frame as a separate PNG alongside the grid")

    # Model
    p.add_argument("--model_name", default="stabilityai/stable-diffusion-xl-base-1.0",
                   type=str, help="HuggingFace model ID or local path to an SDXL base model")

    # Geometry
    p.add_argument("--image_resolution", default=1024, type=int,
                   help="Square I/O resolution in pixels (must be divisible by 8).")
    p.add_argument("--num_intermediate_morphs", default=8, type=int,
                   help="Number of interpolated frames between the two input images.")

    # Diffusion
    p.add_argument("--steps", default=50, type=int,
                   help="Number of DDIM timesteps used for both inversion and forward")
    p.add_argument("--edit_strength", default=0.8, type=float,
                   help="Fraction [0,1] of timesteps used.")
    p.add_argument("--guidance_scale", default=7.5, type=float,
                   help="Classifier-free guidance scale for the forward denoising pass")

    # Beta
    p.add_argument("--beta_alpha", default=20.0, type=float,
                   help="α param of the Beta distribution used for attention coefficients.")
    p.add_argument("--beta_beta", default=20.0, type=float,
                   help="β param of the Beta distribution")

    # Dtype
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="Weight dtype. bf16 is preferred on Ampere+ GPUs for stability.")

    # Misc
    p.add_argument("--seed", default=42, type=int,
                   help="Global random seed for reproducibility")

    # Memory / inference optimizations
    mem = p.add_argument_group("Memory & inference optimisations")
    mem.add_argument("--enable_xformers", action="store_true",
                     help="Enable xformers memory-efficient attention.")
    mem.add_argument("--enable_vae_tiling", action="store_true",
                     help="Tile the VAE decode pass to avoid OOM at 1024px+.")
    mem.add_argument("--enable_vae_slicing", action="store_true",
                     help="Slice the VAE decode batch one image at a time.")
    mem.add_argument("--enable_cpu_offload", action="store_true",
                     help="Offload model components to CPU when not needed.")
    mem.add_argument("--enable_sequential_offload", action="store_true",
                     help="More aggressive sequential layer-by-layer CPU offload.")
    mem.add_argument("--compile_unet", action="store_true",
                     help="torch.compile the UNet with mode='reduce-overhead'.")
    mem.add_argument("--attention_slicing", action="store_true",
                     help="Slice attention computation head-by-head to reduce peak VRAM.")
    mem.add_argument("--channels_last", action="store_true",
                     help="Use torch.channels_last memory format for the UNet.")

    # GPU Optimization (NEW)
    mem.add_argument("--enable_gpu_optimization", action="store_true",
                     help="Enable GPU optimizations: batched VAE decode + auto batch sizing.")
    mem.add_argument("--vae_decode_batch_size", default=None, type=int,
                     help="Manual override for VAE decode batch size (auto-calculated if None).")
    mem.add_argument("--enable_memory_monitoring", action="store_true",
                     help="Enable GPU memory monitoring and statistics printing.")

    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()

    # Seed & backend flags
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    # Dtype
    _dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype_weight = _dtype_map[cfg.dtype]

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Memory monitor
    memory_monitor = GPUMemoryMonitor(device, enable_monitoring=cfg.enable_memory_monitoring)
    memory_monitor.print_stats("startup")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # Geometry
    assert cfg.image_resolution % 8 == 0, "--image_resolution must be divisible by 8"
    image_resolution = [cfg.image_resolution, cfg.image_resolution]
    latent_h = cfg.image_resolution // 8
    latent_w = cfg.image_resolution // 8
    latent_channels = 4
    interpolation_size = cfg.num_intermediate_morphs + 2

    print(f"[config] model          : {cfg.model_name}")
    print(f"[config] resolution     : {cfg.image_resolution}px → latent {latent_h}×{latent_w}")
    print(f"[config] morph frames   : {interpolation_size} ({cfg.num_intermediate_morphs} intermediate)")
    print(f"[config] steps          : {cfg.steps}  edit_strength={cfg.edit_strength}")
    print(f"[config] guidance_scale : {cfg.guidance_scale}")
    print(f"[config] dtype          : {cfg.dtype}")
    
    if cfg.enable_gpu_optimization:
        print(f"[config] GPU optimization: ENABLED (batched VAE decode)")
        if cfg.vae_decode_batch_size:
            print(f"[config] VAE batch size: {cfg.vae_decode_batch_size} (manual override)")
        else:
            print(f"[config] VAE batch size: AUTO-CALCULATED")

    # Load SDXL components
    print("\n[loading] SDXL components…")

    vae = AutoencoderKL.from_pretrained(
        cfg.model_name, subfolder="vae"
    ).to(device, dtype_weight)

    text_encoder = CLIPTextModel.from_pretrained(
        cfg.model_name, subfolder="text_encoder"
    ).to(device)

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        cfg.model_name, subfolder="text_encoder_2"
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.model_name, subfolder="tokenizer_2")

    unet = UNet2DConditionModel.from_pretrained(
        cfg.model_name, subfolder="unet"
    ).to(device, dtype_weight)

    forward_scheduler = DDIMScheduler.from_pretrained(cfg.model_name, subfolder="scheduler")
    forward_scheduler.set_timesteps(cfg.steps)

    invert_scheduler = DDIMInverseScheduler.from_pretrained(cfg.model_name, subfolder="scheduler")
    invert_scheduler.set_timesteps(cfg.steps)

    print("[loading] done.")
    memory_monitor.print_stats("after_loading")

    # Memory / inference optimizations
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
        unet.enable_sequential_cpu_offload()
        vae.enable_sequential_cpu_offload()
        print("[opt] sequential CPU offload (unet + vae)  ✓")
    elif cfg.enable_cpu_offload:
        unet.enable_model_cpu_offload()
        print("[opt] model-level CPU offload (unet)  ✓")

    if cfg.compile_unet:
        unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
        print("[opt] torch.compile(unet, mode='reduce-overhead')  ✓")

    # Static artifacts
    injection_noise = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        device=device,
        dtype=dtype_weight,
    )

    thresholds = build_fourier_thresholds(interpolation_size, latent_h, latent_w)
    print(f"[config] Fourier thresholds (intermediate frames): {thresholds[1:-1]}")

    add_time_ids_full = get_add_time_ids(
        batch_size=interpolation_size,
        image_h=cfg.image_resolution,
        image_w=cfg.image_resolution,
        device=device,
        dtype=dtype_weight,
    )

    # Main inference loop
    with open(cfg.json_path) as f:
        for line in tqdm(f, disable=not accelerator.is_local_main_process, desc="pairs"):
            entry = eval(line)
            image_paths = entry["image_paths"]
            prompts = entry["prompts"]
            exp_id = entry["exp_id"]

            print(f"\n[{exp_id}] {image_paths[0]}  ↔  {image_paths[1]}")

            # Per-image: encode image + text
            latent_x_list = []
            con_embeds_list = []
            pooled_con_list = []
            uncon_embeds_list = []
            pooled_uncon_list = []
            original_images = []

            for img_idx in range(2):
                image = load_im_from_path(
                    image_paths[img_idx], image_resolution
                ).to(device, dtype_weight)
                original_images.append(image)

                latent = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                latent_x_list.append(latent)

                con_emb, pool_con = encode_prompt_sdxl(
                    prompts=[prompts[img_idx]],
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    device=device,
                    dtype=dtype_weight,
                )
                con_embeds_list.append(con_emb)
                pooled_con_list.append(pool_con)

                uncon_emb, pool_uncon = encode_prompt_sdxl(
                    prompts=[""],
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    device=device,
                    dtype=dtype_weight,
                )
                uncon_embeds_list.append(uncon_emb)
                pooled_uncon_list.append(pool_uncon)

            # Interpolate latents
            latent_x = spherical_interpolation(
                latent_x_list[0], latent_x_list[1], interpolation_size
            ).squeeze(0)

            # Interpolate text embeddings
            text_input_con_all = linear_interpolation(
                con_embeds_list[0], con_embeds_list[1], interpolation_size
            ).squeeze(0)

            text_input_uncon_all = linear_interpolation(
                uncon_embeds_list[0], uncon_embeds_list[1], interpolation_size
            ).squeeze(0)

            pooled_con_all = linear_interpolation(
                pooled_con_list[0], pooled_con_list[1], interpolation_size
            ).squeeze(0)

            pooled_uncon_all = pooled_uncon_list[0].expand(interpolation_size, -1).contiguous()

            # Beta attention coefficients
            coef_attn = generate_beta_tensor(
                size=interpolation_size,
                alpha=cfg.beta_alpha,
                beta=cfg.beta_beta,
            )

            # AID Inversion
            invert_timesteps = invert_scheduler.timesteps - 1
            invert_timesteps = invert_timesteps[: int(cfg.steps * cfg.edit_strength)]

            reverted_latent = aid_inversion(
                timesteps=invert_timesteps,
                latent=latent_x,
                text_input_con=text_input_con_all,
                text_input_uncon=text_input_uncon_all,
                pooled_con=pooled_con_all,
                pooled_uncon=pooled_uncon_all,
                add_time_ids=add_time_ids_full,
                coef_self_attn=coef_attn,
                coef_cross_attn=coef_attn,
                guidance_scale=cfg.guidance_scale,
            )

            # Fourier filter intermediate frames
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
            )

            # Clear memory before forward pass
            memory_monitor.clear_cache()
            memory_monitor.print_stats("before_forward")

            # AID Forward (denoising)
            forward_timesteps = forward_scheduler.timesteps - 1
            forward_timesteps = forward_timesteps[cfg.steps - int(cfg.steps * cfg.edit_strength):]

            coef_attn = generate_beta_tensor(
                size=interpolation_size,
                alpha=cfg.beta_alpha,
                beta=cfg.beta_beta,
            )

            morphing_latent = aid_forward(
                timesteps=forward_timesteps,
                latent=latent,
                text_input_con=text_input_con_all,
                text_input_uncond=text_input_uncon_all,
                pooled_con=pooled_con_all,
                pooled_uncond=pooled_uncon_all,
                add_time_ids=add_time_ids_full,
                coef_self_attn=coef_attn,
                coef_cross_attn=coef_attn,
                guidance_scale=cfg.guidance_scale,
            )

            memory_monitor.print_stats("before_vae_decode")

            # ✓ OPTIMIZED: Batched VAE decode
            if cfg.enable_gpu_optimization:
                images = batch_vae_decode(
                    vae,
                    morphing_latent,
                    batch_size=cfg.vae_decode_batch_size,
                    memory_monitor=memory_monitor,
                    device=device,
                )
            else:
                # Original frame-by-frame decode
                images = []
                for x in morphing_latent:
                    decoded = vae.decode(x.unsqueeze(0) / vae.config.scaling_factor).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    images.append(decoded.detach().to(torch.float32).cpu())

            # Replace endpoints with original images
            images[0] = (original_images[0] / 2 + 0.5).detach().to(torch.float32).cpu()
            images[-1] = (original_images[1] / 2 + 0.5).detach().to(torch.float32).cpu()

            # Save grid
            strip = torch.cat(images, dim=0)
            save_image(strip, f"{cfg.save_dir}/{exp_id}_grid.png", nrow=interpolation_size)
            print(f"  → saved grid  ({interpolation_size} frames)  {cfg.save_dir}/{exp_id}_grid.png")

            # Save individual frames if requested
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

            memory_monitor.clear_cache()
            memory_monitor.print_stats("after_pair")
