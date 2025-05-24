import typing as tp
import torch

from .patcher import NoiseAttentionPatcher
from PIL import Image

# https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/stable_diffusion_diffedit/pipeline_stable_diffusion_diffedit.py#L1046
def binarize_noise_based(noise: torch.Tensor, mask_thresholding_ratio: float):
    clamp_magnitude = noise.mean() * mask_thresholding_ratio
    semantic_mask_image = noise.clamp(0, clamp_magnitude) / clamp_magnitude
    semantic_mask_image = torch.where(semantic_mask_image <= 0.5, 0, 1)
    pred_mask = semantic_mask_image.byte()
    return pred_mask


def get_map_noise_based(
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
        out_size: tuple[int, int] | None = None,
) -> torch.Tensor:

    assert hasattr(pipe, "call_patched"), "The pipeline should be patched firstly using `patcher.patch_call(pipe)`."
    patcher.clear()

    generator = torch.Generator().manual_seed(seed)

    pipe.step_and_record(
        prompt_orig=source_prompt,
        prompt_edit=target_prompt,
        num_repeat=num_repeat,
        image=image.resize((512, 512)),
        strength=strength,
        num_inference_steps=num_steps,
        sample_mode=sample_mode,
        generator=generator,
        guidance_scale=0.,
    )
    fm = patcher["noise"][0]
    fm = (fm[::2] - fm[1::2]).mean(dim=(0, 1))

    fm = torch.nn.functional.interpolate(
        input=fm[None, None],
        size=image.size if out_size is None else out_size,
        mode="bilinear",
    ).squeeze()

    return fm
