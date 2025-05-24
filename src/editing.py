import typing as tp

from image_editing_mask.attn_based import get_map_attn_based
from image_editing_mask.common_utils import binarize
from image_editing_mask.noise_based import binarize_noise_based, get_map_noise_based
from image_editing_mask.ours import get_map_ours
from image_editing_mask.patcher import NoiseAttentionPatcher
from image_editing_mask.turbo_edit.main import load_pipe, run
from PIL import Image


MAPPING = {
    "ours": (get_map_ours, binarize),
    "attn_based": (get_map_attn_based, binarize),
    "noise_based": (get_map_noise_based, binarize_noise_based),
}


class EditingPipeline:
    def __init__(self, cache_dir: str):
        self.pipeline, self.patcher = load_and_patch_pipe(cache_dir)

    def edit(
            self,
            image: Image.Image,
            source_prompt: str,
            target_prompt: str,
            mask_obtaining: tp.Literal["ours", "attn_based", "noise_based"],
            binarization_threshold: float,
            num_repeat: int = 1,
            strength: float = 0.5,
            sample_mode: tp.Literal["sample", "argmax"] = "argmax",
            seed: int = 8128,
    ) -> Image.Image:

        return edit_image(
            self.pipeline,
            self.patcher,
            image=image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            mask_obtaining=mask_obtaining,
            binarization_threshold=binarization_threshold,
            num_repeat=num_repeat,
            strength=strength,
            sample_mode=sample_mode,
            seed=seed,
        )
        


def load_and_patch_pipe(cache_dir: str) -> tuple[tp.Any, NoiseAttentionPatcher]:
    pipeline = load_pipe(fp16=True, cache_dir=cache_dir)

    patcher = NoiseAttentionPatcher()
    patcher.patch_attention(pipeline.unet, patch_cross=True, patch_self=False)
    patcher.patch_call(pipeline)

    return pipeline, patcher


def edit_image(
        pipeline,
        patcher: NoiseAttentionPatcher,
        image: Image.Image,
        source_prompt: str,
        target_prompt: str,
        mask_obtaining: tp.Literal["ours", "attn_based", "noise_based"],
        binarization_threshold: float,
        num_repeat: int = 1,
        strength: float = 0.5,
        sample_mode: tp.Literal["sample", "argmax"] = "argmax",
        seed: int = 8128,
) -> Image.Image:

    mask_obtaining_func, binarization_func = MAPPING[mask_obtaining]
    fm = mask_obtaining_func(
        pipeline,
        patcher,
        image,
        source_prompt,
        target_prompt,
        num_repeat=num_repeat,
        strength=strength,
        sample_mode=sample_mode,
        seed=seed,
        out_size=(64, 64)
    )
    mask = binarization_func(fm, binarization_threshold)

    result = run(
        image=image,
        src_prompt=source_prompt,
        tgt_prompt=target_prompt,
        seed=seed,
        w1=1.5,
        num_timesteps=4,
        pipeline=pipeline,
        mask=mask
    )
    return result
