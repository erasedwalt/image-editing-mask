import typing as tp

import torch
from PIL import Image

from .common_utils import (
    encode_prompt,
    fix_timesteps_for_denoising_start_end,
    handle_prompt_diffs,
    high_level_get_timesteps,
    infer_unet,
    prepare_inputs,
    prepare_latents,
    randn_tensor,
    rescale,
)
from .patcher import NoiseAttentionPatcher


def aggregate_cross_attn(
        patcher: NoiseAttentionPatcher,
        changed_slices: list[slice],
        size: int,
) -> torch.Tensor:

    res = torch.tensor(0., device="cpu")
    for key in patcher:
        if key.endswith("attn1"):
            continue

        t = patcher[key][0]

        if t.shape[-1] == 64:
            continue

        if t.shape[-2] != size:
            continue

        d = t.mean(dim=0)

        cat = []
        for slice_ in changed_slices:
            cat.append(d[..., slice_])
        d = torch.cat(cat, dim=-1)

        res = res + d.mean(dim=0).mean(dim=-1)
    return res


@torch.inference_mode()
def _get_maps(
        pipe,
        patcher: NoiseAttentionPatcher,
        image: Image.Image,
        prompt: str,
        changed_slices: list[slice],
        batch_size: int = 2,
        num_images_per_prompt: int = 1,
        strength: float=0.5,
        num_steps: int = 4,
        seed: int = 8128,
        sample_mode: tp.Literal["sample", "argmax"] = "sample",
) -> torch.Tensor:

    patcher.clear()

    denoising_start = None
    denoising_end = None
    generator = torch.Generator().manual_seed(seed)

    processed_image = pipe.image_processor.preprocess([image])
    timesteps, latent_timestep, num_inference_steps, latent_timestep, add_noise = high_level_get_timesteps(
        pipe = pipe,
        num_steps = num_steps,
        batch_size = batch_size,
        num_images_per_prompt = num_images_per_prompt,
        strength = strength,
        denoising_start = denoising_start
    )

    latents = prepare_latents(
        vae=pipe.vae,
        scheduler=pipe.scheduler,
        image=processed_image,
        timestep=latent_timestep,
        batch_size=batch_size,
        num_images_per_prompt=num_images_per_prompt,
        dtype=pipe.dtype,
        device=pipe.device,
        sample_mode=sample_mode,
        generator=generator,
        add_noise=add_noise,
        randn_func=randn_tensor
    )

    height, width = latents.shape[-2:]
    height = height * pipe.vae_scale_factor
    width = width * pipe.vae_scale_factor
    original_size = (height, width)
    target_size = (height, width)

    (
        prompt_embeds,
        _,
        pooled_prompt_embeds,
        _,
    ) = encode_prompt(
        prompt=prompt,
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=None,
        device=pipe.device,
        num_images_per_prompt=num_images_per_prompt
    )

    prompt_embeds, add_text_embeds, add_time_ids = prepare_inputs(
        pipe = pipe,
        batch_size = batch_size,
        num_images_per_prompt = num_images_per_prompt,
        prompt_embeds = prompt_embeds,
        pooled_prompt_embeds = pooled_prompt_embeds,
        original_size = original_size,
        target_size = target_size
    )

    timesteps, num_inference_steps = fix_timesteps_for_denoising_start_end(
        pipe = pipe,
        denoising_start = denoising_start,
        denoising_end = denoising_end,
        timesteps = timesteps,
        num_inference_steps = num_inference_steps
    )

    t = timesteps[0]
    latent_model_input = latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    # predict the noise residual
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    _ = infer_unet(
        unet=pipe.unet,
        latent_model_input=latent_model_input,
        t=t,
        prompt_embeds=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs
    )

    map = aggregate_cross_attn(patcher, changed_slices, 256)
    return map


def get_map_attn_based(
        pipe,
        patcher: NoiseAttentionPatcher,
        image: Image.Image,
        source_prompt: str,
        target_prompt: str,
        num_repeat: int = 1,
        strength: float = 0.5,
        num_steps: int = 4,
        seed: int = 8128,
        sample_mode: tp.Literal["sample", "argmax"] = "sample",
) -> torch.Tensor:

    assert hasattr(pipe.unet, "attn_patched"), "The pipeline should be patched firstly using `patcher.patch_attention(pipe.unet, ...)`."
    patcher.clear()

    prompt, changed_slices, _, _ = handle_prompt_diffs(pipe.tokenizer, source_prompt, target_prompt)

    fm = _get_maps(
        pipe=pipe,
        patcher=patcher,
        image=image,
        prompt=prompt,
        changed_slices=changed_slices,
        batch_size=1,
        num_images_per_prompt=num_repeat,
        strength=strength,
        num_steps=num_steps,
        seed=seed,
        sample_mode=sample_mode,
    )

    fm = fm.reshape(16, 16)
    fm = torch.nn.functional.interpolate(
        input=rescale(fm)[None, None],
        size=image.size,
        mode="bilinear",
    ).squeeze()

    return fm
