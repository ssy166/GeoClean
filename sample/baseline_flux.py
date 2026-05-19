import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


PAPER_BASELINES = [
    "base",
    "np",
    "ca",
    "dev",
    "sld",
    "geoclean",
]

METHOD_ALIASES = {
    "flux": "base",
    "flux.1-dev": "base",
    "negative_prompt": "np",
    "negative-prompt": "np",
    "dve": "dev",
    "ours": "geoclean",
}


@dataclass
class MethodResult:
    images: list
    adapter_note: str


def normalize_method(name: str) -> str:
    key = name.strip().lower()
    return METHOD_ALIASES.get(key, key)


def sanitize_name(value: str, default: str = "prompt") -> str:
    value = str(value)
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:80] if cleaned else default


def ensure_local_scheduler_config(model_path: str) -> None:
    scheduler_dir = os.path.join(model_path, "scheduler")
    config_path = os.path.join(scheduler_dir, "config.json")
    scheduler_config_path = os.path.join(scheduler_dir, "scheduler_config.json")
    if os.path.isfile(config_path) and not os.path.isfile(scheduler_config_path):
        import shutil

        shutil.copyfile(config_path, scheduler_config_path)
        print(f"Created missing scheduler_config.json from config.json: {scheduler_config_path}")


def load_pipeline(model_path: str, device: str, dtype: torch.dtype, local_files_only: bool):
    from diffusers import FluxPipeline

    if os.path.isdir(model_path):
        ensure_local_scheduler_config(model_path)

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def encode_prompts(pipe, prompts: list[str], device: str):
    return pipe.encode_prompt(prompt=prompts, prompt_2=None, device=device)


def make_generator(device: str, seed: int | None):
    if seed is None or seed < 0:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def prepare_flux_latents(
    pipe,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: str,
    seed: int | None,
):
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=make_generator(device, seed),
    )
    latent_image_ids = None
    if isinstance(latents, tuple):
        if len(latents) >= 2:
            latent_image_ids = latents[1]
        latents = latents[0]
    return latents, latent_image_ids


def get_flux_timesteps(pipe, num_inference_steps: int, height: int, width: int, latent_seq_len: int, device: str):
    mu = None
    use_dynamic_shifting = getattr(pipe.scheduler.config, "use_dynamic_shifting", False)
    if use_dynamic_shifting:
        if hasattr(pipe, "calculate_shift"):
            try:
                mu = pipe.calculate_shift(
                    image_seq_len=latent_seq_len,
                    base_seq_len=pipe.scheduler.config.base_image_seq_len,
                    max_seq_len=pipe.scheduler.config.max_image_seq_len,
                    base_shift=pipe.scheduler.config.base_shift,
                    max_shift=pipe.scheduler.config.max_shift,
                )
            except TypeError:
                mu = pipe.calculate_shift(
                    latent_seq_len,
                    pipe.scheduler.config.base_image_seq_len,
                    pipe.scheduler.config.max_image_seq_len,
                    pipe.scheduler.config.base_shift,
                    pipe.scheduler.config.max_shift,
                )
            except Exception:
                mu = None
        if mu is None:
            base_seq_len = getattr(pipe.scheduler.config, "base_image_seq_len", 256)
            max_seq_len = getattr(pipe.scheduler.config, "max_image_seq_len", 4096)
            base_shift = getattr(pipe.scheduler.config, "base_shift", 0.5)
            max_shift = getattr(pipe.scheduler.config, "max_shift", 1.15)
            image_seq_len = (height // 16) * (width // 16)
            mu = base_shift + (max_shift - base_shift) * (image_seq_len - base_seq_len) / (
                max_seq_len - base_seq_len
            )

    if mu is not None:
        pipe.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    return pipe.scheduler.timesteps


def fallback_latent_image_ids(batch_size: int, height: int, width: int, device: str, dtype: torch.dtype):
    vae_scale_factor = 16
    latent_height = 2 * (height // vae_scale_factor)
    latent_width = 2 * (width // vae_scale_factor)
    latent_image_ids = torch.zeros(latent_height // 2, latent_width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(latent_height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(latent_width // 2)[None, :]
    latent_image_ids = latent_image_ids.reshape((latent_height // 2) * (latent_width // 2), 3)
    return latent_image_ids.to(device=device, dtype=dtype)


@torch.no_grad()
def decode_latents(pipe, latents, height: int, width: int):
    vae_scale_factor = 16
    batch_size, _, channels = latents.shape
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    latents = latents.view(batch_size, latent_h, latent_w, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(
        batch_size, channels // 4, latent_h * 2, latent_w * 2
    )
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def transformer_uses_guidance(pipe) -> bool:
    return bool(getattr(pipe.transformer.config, "guidance_embeds", False))


def build_call_model_fn(pipe, img_ids, guidance_scale: float, device: str):
    uses_guidance = transformer_uses_guidance(pipe)

    def call_model(latents_in, t_val, prompt_embs, pooled_embs, txt_ids_in, is_large_timestep):
        t_input = t_val.expand(latents_in.shape[0])
        if is_large_timestep:
            t_input = t_input / 1000.0

        kwargs = dict(
            hidden_states=latents_in,
            timestep=t_input,
            encoder_hidden_states=prompt_embs,
            pooled_projections=pooled_embs,
            img_ids=img_ids,
            txt_ids=txt_ids_in,
            return_dict=False,
        )
        if uses_guidance:
            kwargs["guidance"] = torch.tensor(
                [guidance_scale], device=device, dtype=prompt_embs.dtype
            ).expand(latents_in.shape[0])
        return pipe.transformer(**kwargs)[0]

    return call_model


def tensor_project(source, basis):
    numerator = (source.float() * basis.float()).sum(dim=(1, 2), keepdim=True)
    denominator = (basis.float() * basis.float()).sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return (numerator / denominator).to(source.dtype) * basis


@torch.no_grad()
def run_vector_method(
    pipe,
    prompts: list[str],
    method: str,
    concept: str,
    anchor_concept: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    safety_scale: float,
    sld_threshold: float,
    acs_delta: float,
    neg_cfg_scale: float,
    ca_scale: float,
    dev_strength: float,
    dev_threshold: float,
    height: int,
    width: int,
    seed: int | None,
    device: str,
) -> MethodResult:
    batch_size = len(prompts)
    pos_embeds, pos_pooled, pos_txt_ids = encode_prompts(pipe, prompts, device)
    dtype = pos_embeds.dtype

    latents, latent_image_ids = prepare_flux_latents(
        pipe, batch_size, height, width, dtype=dtype, device=device, seed=seed
    )
    timesteps = get_flux_timesteps(
        pipe,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        latent_seq_len=latents.shape[1],
        device=device,
    )
    img_ids = (
        latent_image_ids.to(device=device, dtype=dtype)
        if latent_image_ids is not None
        else fallback_latent_image_ids(batch_size, height, width, device=device, dtype=dtype)
    )
    call_model = build_call_model_fn(pipe, img_ids, guidance_scale, device)
    is_large_timestep = timesteps[0] > 100
    max_t = 1000.0 if is_large_timestep else 1.0
    threshold_t = sld_threshold * max_t

    concept_text = concept or ""
    negative_text = negative_prompt or concept_text
    anchor_text = anchor_concept or ""

    needs_concept = method in {"np", "ca", "sld", "dev", "geoclean"}
    if needs_concept:
        concept_embeds, concept_pooled, concept_txt_ids = encode_prompts(
            pipe, [concept_text] * batch_size, device
        )
        uncond_embeds, uncond_pooled, uncond_txt_ids = encode_prompts(pipe, [""] * batch_size, device)
    if method == "np":
        neg_embeds, neg_pooled, neg_txt_ids = encode_prompts(pipe, [negative_text] * batch_size, device)
    if method == "dev":
        anchor_embeds, anchor_pooled, anchor_txt_ids = encode_prompts(
            pipe, [anchor_text] * batch_size, device
        )

    u_bar = None
    desc = f"{method} FLUX"
    for i, t in enumerate(tqdm(timesteps, desc=desc)):
        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0.0, device=device)
        dt = prev_t - t
        if is_large_timestep:
            dt = dt / 1000.0

        v_pos = call_model(latents, t, pos_embeds, pos_pooled, pos_txt_ids, is_large_timestep)
        v_final = v_pos
        current_t = t.item() if isinstance(t, torch.Tensor) else float(t)

        if method == "np":
            v_neg = call_model(latents, t, neg_embeds, neg_pooled, neg_txt_ids, is_large_timestep)
            v_final = v_neg + neg_cfg_scale * (v_pos - v_neg)
        elif method == "ca":
            v_concept = call_model(
                latents, t, concept_embeds, concept_pooled, concept_txt_ids, is_large_timestep
            )
            v_uncond = call_model(
                latents, t, uncond_embeds, uncond_pooled, uncond_txt_ids, is_large_timestep
            )
            concept_basis = v_concept - v_uncond
            prompt_direction = v_pos - v_uncond
            v_final = v_pos - ca_scale * tensor_project(prompt_direction, concept_basis)
        elif method == "sld" and safety_scale > 0 and current_t > threshold_t:
            v_concept = call_model(
                latents, t, concept_embeds, concept_pooled, concept_txt_ids, is_large_timestep
            )
            v_uncond = call_model(
                latents, t, uncond_embeds, uncond_pooled, uncond_txt_ids, is_large_timestep
            )
            v_final = v_pos - safety_scale * (v_concept - v_uncond)
        elif method == "dev":
            v_anchor = call_model(
                latents, t, anchor_embeds, anchor_pooled, anchor_txt_ids, is_large_timestep
            )
            v_concept = call_model(
                latents, t, concept_embeds, concept_pooled, concept_txt_ids, is_large_timestep
            )
            delta_v = v_anchor - v_concept
            delta_norm = torch.linalg.vector_norm(delta_v, dim=(1, 2), keepdim=True)
            delta_unit = delta_v / (delta_norm + 1e-8)
            score = (v_pos * delta_unit).sum(dim=(1, 2), keepdim=True)
            correction_mask = (score < dev_threshold).to(dtype=score.dtype)
            correction_mag = correction_mask * dev_strength * (dev_threshold - score)
            v_final = v_pos + correction_mag.to(v_pos.dtype) * delta_unit
        elif method == "geoclean" and safety_scale > 0 and current_t > threshold_t:
            x_anchor = latents + dt * v_pos
            v_concept_anchor = call_model(
                x_anchor, prev_t, concept_embeds, concept_pooled, concept_txt_ids, is_large_timestep
            )
            v_uncond_anchor = call_model(
                x_anchor, prev_t, uncond_embeds, uncond_pooled, uncond_txt_ids, is_large_timestep
            )
            u = safety_scale * (v_concept_anchor - v_uncond_anchor)
            if u_bar is None:
                u_bar = torch.zeros_like(v_pos)
            delta_u = u - u_bar
            delta_norm = torch.linalg.vector_norm(delta_u, dim=(1, 2), keepdim=True)
            pos_norm = torch.linalg.vector_norm(v_pos, dim=(1, 2), keepdim=True)
            limit = acs_delta * pos_norm
            clip_factor = torch.minimum(torch.ones_like(limit), limit / (delta_norm + 1e-6))
            u_bar = u_bar + delta_u * clip_factor
            v_final = v_pos - u_bar

        latents = latents + dt * v_final

    note = {
        "base": "reference FLUX generation",
        "np": "classic negative-prompt vector guidance",
        "ca": "training-free concept-ablation vector projection",
        "dev": "DVE/DEV directional vector erasure adapter",
        "sld": "standard SLD-style safety-vector guidance",
        "geoclean": "CLE + ACS GeoClean sampler",
    }[method]
    return MethodResult(images=decode_latents(pipe, latents, height, width), adapter_note=note)


def load_prompts(csv_path: str, prompt_col: str, num_samples: int) -> list[str]:
    if not csv_path:
        return ["a small red cube on a white table"]
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or prompt_col not in reader.fieldnames:
            raise ValueError(f"Prompt column '{prompt_col}' not found in {csv_path}")
        prompts = [(row.get(prompt_col) or "").strip() for row in reader]
    prompts = [prompt for prompt in prompts if prompt]
    if num_samples > 0:
        prompts = prompts[:num_samples]
    if not prompts:
        raise ValueError(f"No prompts found in {csv_path}")
    return prompts


def save_metadata(output_dir: str, record: dict) -> None:
    path = os.path.join(output_dir, "metadata.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Run paper-table FLUX baselines in one environment.")
    parser.add_argument("--model_path", type=str, default=os.environ.get("GEOCLEAN_MODEL_PATH", "./models/FLUX.1-dev"))
    parser.add_argument("--csv_path", type=str, default="data/i2p_nudity_prompts.csv")
    parser.add_argument("--prompt_col", type=str, default="prompt")
    parser.add_argument("--output_dir", type=str, default="results/baselines_flux")
    parser.add_argument("--methods", nargs="+", default=PAPER_BASELINES)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--concept", type=str, default="nudity")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--anchor_concept", type=str, default="dressed in modest clothing")
    parser.add_argument("--safety_scale", type=float, default=1.5)
    parser.add_argument("--sld_threshold", type=float, default=0.7)
    parser.add_argument("--acs_delta", type=float, default=0.2)
    parser.add_argument("--neg_cfg_scale", type=float, default=3.0)
    parser.add_argument("--ca_scale", type=float, default=1.0)
    parser.add_argument("--dev_strength", type=float, default=1.5)
    parser.add_argument("--dev_threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--allow_remote", action="store_true", help="Allow model download instead of local-only loading.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    methods = [normalize_method(method) for method in args.methods]
    unknown = [method for method in methods if method not in PAPER_BASELINES]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {PAPER_BASELINES}")

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.csv_path, args.prompt_col, args.num_samples)
    pipe = load_pipeline(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        local_files_only=not args.allow_remote,
    )

    vector_methods: dict[str, Callable] = {
        "base": run_vector_method,
        "np": run_vector_method,
        "ca": run_vector_method,
        "dev": run_vector_method,
        "sld": run_vector_method,
        "geoclean": run_vector_method,
    }

    for prompt_index, prompt_text in enumerate(prompts):
        prompt_batch = [prompt_text]
        prompt_tag = sanitize_name("_".join(prompt_text.split()[:6]), default=f"prompt{prompt_index:04d}")
        for method_index, method in enumerate(methods):
            seed = args.seed + prompt_index if args.seed >= 0 else None
            print(f"\n=== method={method} prompt_index={prompt_index} ===")
            if method in vector_methods:
                result = run_vector_method(
                    pipe=pipe,
                    prompts=prompt_batch,
                    method=method,
                    concept=args.concept,
                    anchor_concept=args.anchor_concept,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    safety_scale=args.safety_scale,
                    sld_threshold=args.sld_threshold,
                    acs_delta=args.acs_delta,
                    neg_cfg_scale=args.neg_cfg_scale,
                    ca_scale=args.ca_scale,
                    dev_strength=args.dev_strength,
                    dev_threshold=args.dev_threshold,
                    height=args.height,
                    width=args.width,
                    seed=seed,
                    device=args.device,
                )
            else:
                raise ValueError(f"Unsupported method '{method}'. Supported: {PAPER_BASELINES}")

            for image_index, image in enumerate(result.images):
                filename = f"{prompt_index:04d}_{method_index:02d}_{method}_{prompt_tag}.png"
                save_path = os.path.join(args.output_dir, filename)
                image.save(save_path)
                print(f"Saved: {save_path}")
                save_metadata(
                    args.output_dir,
                    {
                        "method": method,
                        "prompt_index": prompt_index,
                        "image_index": image_index,
                        "prompt": prompt_text,
                        "concept": args.concept,
                        "seed": seed,
                        "path": save_path,
                        "adapter_note": result.adapter_note,
                    },
                )


if __name__ == "__main__":
    main()
