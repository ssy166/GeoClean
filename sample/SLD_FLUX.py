import torch
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import ModelManager

def run_sld_flux_inference(
    pipe,
    prompt: list,
    safety_concept: str = None,
    safety_guidance_scale: float = 2000.0, # Flux needs higher guidance usually? Or maybe not. Let's keep consistent first.
    use_rectified_sld: bool = True,
    sld_early_stop_threshold: float = 0.5, # Stop erasing after t < 0.5 (noise=1 -> image=0)
    rectified_sld_early_stop: float = 1.0, # 1.0 keeps CLE active whenever safety guidance is active.
    sld_momentum_threshold: float = 0.0, # 0.0 means unused by default
    sld_use_norm_rescaling: bool = False,
    sld_norm_alpha: float = 1.0, # Alpha for Scheme A clamping
    sld_use_gtr3: bool = False, # Enable ACS correction smoothing.
    sld_gtr3_delta: float = 0.1, # ACS stepwise deviation threshold.
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5, # Flux default is often 3.5
    legacy_euler_update: bool = False,
    output_path: str = "results/flux_sample.png",
    device: str = "cuda"
):
    """
    Run inference with Safe Latent Diffusion (SLD) logic adapted for FLUX.1-dev.
    """
    
    if isinstance(prompt, str):
        prompt = [prompt]
    
    batch_size = len(prompt)
    
    # --- 1. Encode Prompts ---
    # FLUX uses 2 text encoders (CLIP + T5)
    # We need embeddings for: Positive, Negative (Uncond), and Safety Concept
    
    # Helper to clean up safety concept list to string
    if safety_concept is None:
        # If no safety concept provided, disable safety guidance
        safety_guidance_scale = 0.0
        safety_concept_str = ""
    elif isinstance(safety_concept, list):
        safety_concept_str = ", ".join(safety_concept)
    else:
        safety_concept_str = safety_concept

    print(f"Running FLUX SLD Inference: Steps={num_inference_steps}, CFG={guidance_scale}, SafetyScale={safety_guidance_scale}")
    if safety_guidance_scale > 0:
        print(f"Suppressing: {safety_concept_str[:50]}...")

    # Encode Standard Prompts
    # FLUX pipeline encode_prompt returns dict usually or tuple?
    # pipe.encode_prompt(prompt, prompt_2, device, ...)
    # It returns: (prompt_embeds, pooled_prompt_embeds, text_ids)
    # FLUX doesn't usually use negative prompts in the standard API (guidance > 1 is "distilled guidance" usually, NOT CFG).
    # BUT, FLUX.1-dev supports CFG if we manually handle it or if guidance_scale > 1 in pipe call?
    # Actually, FLUX.1-dev is guidance distilled, but some implementations support CFG.
    # The official diffusers pipeline for Flux does NOT support negative_prompt by default in `__call__` unless we check.
    # Wait, `guidance_scale` in Flux usually refers to the distilled guidance embedding value, NOT CFG scale.
    # So "Classifier-Free Guidance" in the traditional sense (uncond - cond) is NOT how Flux works by default.
    # Flux has a `guidance` parameter (default 3.5) which is an input to the transformer.
    
    # HOWEVER, for SLD to work, we need a "Negative Direction" (Safety Concept).
    # We can try to implement "CFG-like" steering even if the model wasn't trained for it?
    # Or does SLD require us to calculate v_safety?
    # Yes, we need v_safety.
    # So we need to run model with `safety_concept` as prompt.
    
    # Flux inputs:
    # prompt_embeds, pooled_prompt_embeds, text_ids (from CLIP+T5)
    # guidance (vector)
    
    # Let's Encode Positive Prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device
    )
    
    # Encode Safety Concept
    (
        safety_embeds,
        safety_pooled_embeds,
        safety_text_ids,
    ) = pipe.encode_prompt(
        prompt=[safety_concept_str] * batch_size,
        prompt_2=None,
        device=device
    )
    
    # Prepare Latents
    # Flux VAE channels = 16 (usually transformer.in_channels // 4 because of 2x2 patch packing)
    # pipe.transformer.config.in_channels is 64.
    num_channels_latents = pipe.transformer.config.in_channels // 4
    
    latents = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height=1024,
        width=1024,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=None
    )
    
    # Handle tuple return from prepare_latents (possible in some diffusers versions)
    # It might return (latents, latent_image_ids, ...)
    latent_image_ids = None
    if isinstance(latents, tuple):
        if len(latents) >= 2:
            latent_image_ids = latents[1]
        latents = latents[0]

    # Prepare Timesteps (FLUX requires 'mu' for dynamic shifting)
    # Calculate 'mu' based on image size (default Flux 1024x1024)
    height = 1024
    width = 1024
    image_seq_len = (height // 16) * (width // 16)
    
    mu = None
    # Check if dynamic shifting is enabled in scheduler config
    use_dynamic_shifting = getattr(pipe.scheduler.config, "use_dynamic_shifting", False)
    
    if use_dynamic_shifting:
         if hasattr(pipe, "calculate_shift"):
             try:
                 mu = pipe.calculate_shift(
                    image_seq_len=image_seq_len,
                    base_seq_len=pipe.scheduler.config.base_image_seq_len,
                    max_seq_len=pipe.scheduler.config.max_image_seq_len,
                    base_shift=pipe.scheduler.config.base_shift,
                    max_shift=pipe.scheduler.config.max_shift,
                 )
             except Exception as e:
                 print(f"⚠️ pipe.calculate_shift failed: {e}")
                 mu = None

         if mu is None:
             # Fallback manual calculation
             print("⚠️ Calculating 'mu' manually...")
             base_seq_len = getattr(pipe.scheduler.config, "base_image_seq_len", 256)
             max_seq_len = getattr(pipe.scheduler.config, "max_image_seq_len", 4096)
             base_shift = getattr(pipe.scheduler.config, "base_shift", 0.5)
             max_shift = getattr(pipe.scheduler.config, "max_shift", 1.15)
             
             mu = base_shift + (max_shift - base_shift) * (image_seq_len - base_seq_len) / (max_seq_len - base_seq_len)
    
    print(f"🔧 Timestep Config: use_dynamic_shifting={use_dynamic_shifting}, mu={mu}")

    if mu is not None:
        pipe.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        
    timesteps = pipe.scheduler.timesteps
    
    # Prepare Guidance Embedding
    guidance = torch.tensor([guidance_scale], device=device, dtype=prompt_embeds.dtype)
    guidance = guidance.expand(batch_size) 
    
    # Encode Empty Prompt (Uncond)
    (
        uncond_embeds,
        uncond_pooled_embeds,
        uncond_text_ids,
    ) = pipe.encode_prompt(
        prompt=[""] * batch_size,
        prompt_2=None,
        device=device
    )

    # Determine timestep scaling for threshold-related logic only.
    is_large_timestep = timesteps[0] > 100
    
    # Initialize ACS state.
    u_bar = None # Running mean of correction (smooth u)
    
    # Precompute SLD Threshold
    max_t = 1000.0 if is_large_timestep else 1.0
    thresh_val = sld_early_stop_threshold * max_t
    cle_thresh_val = rectified_sld_early_stop * max_t

    # --- Denoising Loop ---
    # Prepare Image IDs (Precompute outside)
    # Prepare Image IDs (Precompute outside)
    if latent_image_ids is None:
        vae_scale_factor = 16
        latent_height = 2 * (height // vae_scale_factor)
        latent_width = 2 * (width // vae_scale_factor)
        
        def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
            latent_image_ids = torch.zeros(height // 2, width // 2, 3)
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

            latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

            latent_image_ids = latent_image_ids.reshape(
                latent_image_id_height * latent_image_id_width, latent_image_id_channels
            )

            return latent_image_ids.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

        img_ids = _prepare_latent_image_ids(batch_size, latent_height, latent_width, device, prompt_embeds.dtype)
    else:
        img_ids = latent_image_ids.to(device=device, dtype=prompt_embeds.dtype)

    # Helper for model call
    def call_model(latents_in, t_val, prompt_embs, pooled_embs, txt_ids_in):
        # Align with official FluxPipeline: transformer expects timestep / 1000
        t_input = t_val.expand(latents_in.shape[0]).to(latents_in.dtype) / 1000.0
        guidance_input = torch.tensor(
            [guidance_scale], device=device, dtype=prompt_embs.dtype
        ).expand(latents_in.shape[0])
        model_img_ids = img_ids
        if model_img_ids.dim() == 3 and model_img_ids.shape[0] != latents_in.shape[0]:
            if latents_in.shape[0] % model_img_ids.shape[0] == 0:
                model_img_ids = model_img_ids.repeat(latents_in.shape[0] // model_img_ids.shape[0], 1, 1)
            else:
                model_img_ids = model_img_ids[:1].expand(latents_in.shape[0], -1, -1)

        return pipe.transformer(
            hidden_states=latents_in,
            timestep=t_input,
            encoder_hidden_states=prompt_embs,
            pooled_projections=pooled_embs,
            img_ids=model_img_ids,
            txt_ids=txt_ids_in,
            guidance=guidance_input, # global guidance scale
            return_dict=False
        )[0]

    def cat_condition_ids(first, second):
        if first.dim() == 2:
            return first
        return torch.cat([first, second], dim=0)

    def paired_model_call(
        latents_in,
        t_val,
        first_embeds,
        first_pooled,
        first_txt_ids,
        second_embeds,
        second_pooled,
        second_txt_ids,
    ):
        paired_latents = torch.cat([latents_in, latents_in], dim=0)
        paired_embeds = torch.cat([first_embeds, second_embeds], dim=0)
        paired_pooled = torch.cat([first_pooled, second_pooled], dim=0)
        paired_txt_ids = cat_condition_ids(first_txt_ids, second_txt_ids)
        paired_output = call_model(
            paired_latents,
            t_val,
            paired_embeds,
            paired_pooled,
            paired_txt_ids,
        )
        return paired_output.chunk(2, dim=0)

    # --- Denoising Loop ---
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(0)

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            # Calculate dt (Euler)
            current_t = t
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else torch.tensor(0.0, device=device)
            dt = prev_t - current_t
            
            # Scale dt if using 1000-scale timesteps
            if is_large_timestep:
                dt = dt / 1000.0

            # v_pos
            noise_pred_pos = call_model(latents, t, prompt_embeds, pooled_prompt_embeds, text_ids)
            
            # Default v_final
            v_final = noise_pred_pos
            
            # Check Early Stopping Condition (t > Threshold)
            current_t_val = t.item() if isinstance(t, torch.Tensor) else t
            
            # SLD active only in EARLY stage (e.g. t > 0.7)
            # thresh_val is precomputed outside loop
            
            if safety_guidance_scale > 0 and current_t_val > thresh_val:
                # Calculate safety vector.
                cle_time_active = rectified_sld_early_stop >= 1.0 or current_t_val > cle_thresh_val
                use_cle_now = use_rectified_sld and cle_time_active
                if use_cle_now:
                    # CLE: evaluate correction at a lookahead anchor.
                    x_anchor = latents + dt * noise_pred_pos

                    # Ensure prev_t is handled correctly for anchor
                    # If prev_t is 0 (end), maybe anchor doesn't make sense? usually fine.
                    noise_pred_safety_anc, noise_pred_uncond_anc = paired_model_call(
                        x_anchor,
                        prev_t,
                        safety_embeds,
                        safety_pooled_embeds,
                        safety_text_ids,
                        uncond_embeds,
                        uncond_pooled_embeds,
                        uncond_text_ids,
                    )

                    safety_vec = noise_pred_safety_anc - noise_pred_uncond_anc
                else:
                    # Standard SLD: Use current t predictions
                    noise_pred_safety, noise_pred_uncond = paired_model_call(
                        latents,
                        t,
                        safety_embeds,
                        safety_pooled_embeds,
                        safety_text_ids,
                        uncond_embeds,
                        uncond_pooled_embeds,
                        uncond_text_ids,
                    )

                    safety_vec = noise_pred_safety - noise_pred_uncond
                
                # Apply Guidance & Norm Rescaling
                # --- Correction Logic Consolidated ---
                # 1. Calculate Raw Correction
                u = safety_guidance_scale * safety_vec
                
                # 2. Apply Norm Rescaling (Scheme A) if enabled
                if sld_use_norm_rescaling:
                    u_norm = torch.linalg.vector_norm(u, dim=(1,2), keepdim=True)
                    pos_norm = torch.linalg.vector_norm(noise_pred_pos, dim=(1,2), keepdim=True)
                    
                    # Clamp factor: min(1, alpha * |v_pos| / |u|)
                    limit = sld_norm_alpha * pos_norm
                    factor = torch.minimum(torch.ones_like(limit), limit / (u_norm + 1e-6))
                    u = u * factor # Update u in-place for subsequent steps
                
                # 3. Apply ACS correction smoothing if enabled
                if sld_use_gtr3:
                    if u_bar is None:
                        u_bar = torch.zeros_like(noise_pred_pos)
                        
                    # Calculate Delta
                    # delta_u = u_current - u_bar_prev
                    delta_u = u - u_bar
                    
                    # Calculate Norms
                    delta_norm = torch.linalg.vector_norm(delta_u, dim=(1,2), keepdim=True)
                    pos_norm = torch.linalg.vector_norm(noise_pred_pos, dim=(1,2), keepdim=True)
                    
                    # Clip Delta
                    # limit = delta * |v_pos|
                    limit = sld_gtr3_delta * pos_norm
                    clip_factor = torch.minimum(torch.ones_like(limit), limit / (delta_norm + 1e-6))
                    
                    delta_u_clipped = delta_u * clip_factor
                    
                    # Update Running Mean
                    u_bar = u_bar + delta_u_clipped
                    
                    # Final Correction using smoothed u_bar
                    v_final = noise_pred_pos - u_bar
                else:
                    v_final = noise_pred_pos - u

            # Step
            if legacy_euler_update:
                # Legacy manual Euler update (old script behavior)
                latents = latents + dt * v_final
            else:
                # Official scheduler update for better compatibility across versions
                latents = pipe.scheduler.step(v_final, t, latents, return_dict=False)[0]

    # --- Decode ---
    # Unpack latents logic handled by image_processor or manual?
    # Flux VAE expects unpacked latents usually?
    # pipe.decode_latents(latents, ...) handling.
    # We should let the pipeline handle decoding if possible, or replicate unpacking.
    
    # Official FluxPipeline `__call__` uses `self._unpack_latents` then VAE decode.
    
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = height // vae_scale_factor
        width = width // vae_scale_factor
        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels // 4, height * 2, width * 2)
        return latents

    # Flux VAE scale factor is usually 16? Or 8?
    # transformer patch size 2?
    # Let's rely on standard unpacking.
    
    latents = _unpack_latents(latents, 1024, 1024, 16) # Standard Flux VAE scale?
    # Actually, let's look at `pipe.vae_scale_factor`.
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1) if hasattr(pipe, "vae") else 16
    
    # But wait, `_unpack_latents` logic above assumes specific packing.
    # Safest is to call `pipe.image_processor.postprocess`? No, that expects decoded pixel images.
    
    # Let's just create a dummy pipe call? No.
    # We decode manually using VAE.
    
    with torch.no_grad():
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
    
    image = pipe.image_processor.postprocess(image, output_type="pil")
    
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLD Inference with FLUX")
    
    # Dataset Config
    parser.add_argument("--csv_path", type=str, default="data/coco_100/coco_prompts.csv", help="Path to input CSV")
    parser.add_argument("--prompt_col", type=str, default="prompt", help="Column name for prompts")
    parser.add_argument("--safety_concept", type=str, default=None, help="Safety concept")
    
    # Output Config
    parser.add_argument("--output_dir", type=str, default="results/flux_sld", help="Output directory")
    
    # SLD Config
    parser.add_argument("--use_sld", action="store_true", default=False, help="Enable safety guidance")
    parser.add_argument("--use_rectified_sld", "--use_cle", action="store_true", dest="use_rectified_sld", default=False,
                        help="Enable Competition-Aware Lookahead Evaluation (CLE)")
    parser.add_argument("--safety_scale", type=float, default=0, help="Safety guidance scale (High for Flux?)")
    
    # Adaptive SLD Config
    parser.add_argument("--sld_early_stop", type=float, default=0.7, help="Global safety-guidance early-stop threshold")
    parser.add_argument("--rectified_sld_early_stop", "--cle_early_stop", type=float, default=1.0,
                        help="CLE early-stop threshold; 1.0 keeps CLE active whenever safety guidance is active")
    parser.add_argument("--sld_use_norm_rescaling", action="store_true", default=False, help="Enable Norm Rescaling (Scheme A)")
    parser.add_argument("--sld_norm_alpha", type=float, default=0.5, help="Alpha for Scheme A clamping")
    parser.add_argument("--sld_use_gtr3", "--use_acs", action="store_true", dest="sld_use_gtr3", default=False,
                        help="Enable Amplification-Controlled Correction Smoothing (ACS)")
    parser.add_argument("--sld_gtr3_delta", "--acs_delta", type=float, dest="sld_gtr3_delta", default=0.2,
                        help="ACS stepwise deviation threshold")
    
    # General Config
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (Flux is heavy)")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale (Distilled)")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Steps")
    parser.add_argument("--legacy_euler_update", action="store_true", default=False,
                        help="Use legacy manual Euler latent update instead of scheduler.step")
    args = parser.parse_args()

    # Default Flux Safety Concepts (User provided)
    DEFAULT_SAFETY_CONCEPTS = [
        'hate', 'harassment', 'violence', 'suffering', 'humiliation', 'harm', 'suicide',
        'sexual', 'nudity', 'bodily fluids', 'blood', 'obscene gestures', 'illegal activity',
        'drug use', 'theft', 'vandalism', 'weapons', 'child abuse', 'brutality', 'cruelty'
    ]

    # --- Configuration ---
    DATASET_CONFIG = {
        "path": args.csv_path,
        "prompt_col": args.prompt_col,
        "safety_concept": [s.strip() for s in args.safety_concept.split(',')] if args.safety_concept else DEFAULT_SAFETY_CONCEPTS
    }
    
    CSV_PATH = DATASET_CONFIG["path"]
    
    # Auto-generate Output Directory Name
    # Format: {Dataset}_{CFG}_{SLD}_{Re}_{Norm}
    if args.output_dir == "results/flux_sld": # Only override if default
        dataset_name = os.path.splitext(os.path.basename(CSV_PATH))[0] # Full name
        cfg_str = f"cfg{args.guidance_scale}"
        sld_str = f"sld{args.safety_scale}"
        cle_str = "CLE1" if args.use_rectified_sld else "CLE0"
        norm_str = f"Norm{args.sld_norm_alpha}" if args.sld_use_norm_rescaling else "NormOff"
        acs_str = f"ACS-{args.sld_gtr3_delta}" if args.sld_use_gtr3 else "ACSOff"
        
        dir_name = f"{dataset_name}_{cfg_str}_{sld_str}_{cle_str}_{norm_str}_{acs_str}"
        OUTPUT_DIR = os.path.join("results", dir_name)
    else:
        OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    
    # Experiment toggles
    USE_SLD = args.use_sld
    USE_RECTIFIED_SLD = args.use_rectified_sld
    SAFETY_SCALE = args.safety_scale if USE_SLD else 0.0 
    
    # Adaptive SLD Config
    SLD_EARLY_STOP = args.sld_early_stop
    RECTIFIED_SLD_EARLY_STOP = args.rectified_sld_early_stop
    SLD_USE_NORM_RESCALING = args.sld_use_norm_rescaling
    SLD_NORM_ALPHA = args.sld_norm_alpha
    SLD_USE_GTR3 = args.sld_use_gtr3
    SLD_GTR3_DELTA = args.sld_gtr3_delta
    
    GUIDANCE_SCALE = args.guidance_scale
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    # Load Prompts
    try:
        df = pd.read_csv(CSV_PATH)
        all_prompts = df[DATASET_CONFIG["prompt_col"]].tolist()
        print(f"Loaded {len(all_prompts)} prompts.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        all_prompts = ["A photo of a cat"]
    
    # Load Model
    print("Loading Model (FLUX.1-dev)...")
    model_path = "./models/FLUX.1-dev"
    manager = ModelManager(model_id=model_path, device="cuda")
    pipe = manager.load_pipeline()
    
    # Run
    total_prompts = len(all_prompts)
    method_tag = "geoclean" if USE_RECTIFIED_SLD and SLD_USE_GTR3 else ("cle" if USE_RECTIFIED_SLD else "standard")
    
    for i in range(0, total_prompts, BATCH_SIZE):
        batch_prompts = all_prompts[i : i + BATCH_SIZE]
            
        try:
            images = run_sld_flux_inference(
                pipe=pipe,
                prompt=batch_prompts,
                safety_concept=DATASET_CONFIG["safety_concept"],
                safety_guidance_scale=SAFETY_SCALE,
                use_rectified_sld=USE_RECTIFIED_SLD,
                sld_early_stop_threshold=SLD_EARLY_STOP,
                rectified_sld_early_stop=RECTIFIED_SLD_EARLY_STOP,
                sld_use_norm_rescaling=SLD_USE_NORM_RESCALING,
                sld_norm_alpha=SLD_NORM_ALPHA,
                sld_use_gtr3=SLD_USE_GTR3,
                sld_gtr3_delta=SLD_GTR3_DELTA,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=GUIDANCE_SCALE,
                legacy_euler_update=args.legacy_euler_update
            )
            
            for local_idx, image in enumerate(images):
                global_idx = i + local_idx
                filename = f"{global_idx:04d}_cfg{GUIDANCE_SCALE}_sld{SAFETY_SCALE}_{method_tag}.png"
                save_path = os.path.join(OUTPUT_DIR, filename)
                image.save(save_path)
                print(f"Saved: {save_path}")
                
        except OSError as e:
            if e.errno == 28:  # No space left on device — fatal, abort
                print(f"FATAL: Disk full, aborting run: {e}")
                import sys; sys.exit(1)
            print(f"Error (OSError): {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

# Define cls_timestep_scale because it was referenced but not defined in function scope
# Used in transformer calls
cls_timestep_scale = False # Flux usually doesn't need explicit div 1000 if using scheduler sigmas which are 0..1?
# Wait, FlowMatchEulerDiscreteScheduler (default for Flux) uses sigmas 1.0 -> 0.0.
# The transformer usually expects these. So no /1000 needed if they are already 1.0.
# Adjust logic in loop.
"""
python sample/SLD_FLUX.py \
  --csv_path data/i2p_subset.csv \
  --output_dir results/flux_sld \
  --safety_scale 20.0 \
  --use_rectified_sld \
  --sld_use_norm_rescaling \
  --sld_use_gtr3 \
  --sld_gtr3_delta 0.1
"""
