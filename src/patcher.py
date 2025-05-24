import gc
import math
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    MultiPipelineCallbacks,
    PipelineCallback,
    PipelineImageInput,
    deprecate,
    randn_tensor,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_latents,
)


def patched_randn_tensor(shape: Tuple | List, *args, **kwargs) -> torch.Tensor:
    bsz = shape[0]
    assert bsz % 2 == 0
    new_shape = (bsz // 2, *shape[1:])
    noise = randn_tensor(new_shape, *args, **kwargs)
    return torch.repeat_interleave(noise, repeats=2, dim=0)


class BasePatcher(ABC, dict[str | int, list[torch.Tensor]]):
    def clear_activations(self):
        keys = list(self.keys())
        for key in keys:
            self.pop(key)

        gc.collect()
        torch.cuda.empty_cache()


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    scores_copy = attn_weight.clone()
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, scores_copy


class NoiseAttentionPatcher(BasePatcher):
    def patch_attention(patcher_self, unet: torch.nn.Module, patch_cross: bool = True, patch_self: bool = False) -> None:
        def get_attn_call(attn: torch.nn.Module, attn_name: str):
            def new_call(
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                temb: torch.Tensor | None = None,
                *args,
                **kwargs,
            ) -> torch.Tensor:

                residual = hidden_states
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    # scaled_dot_product_attention expects attention_mask shape to be
                    # (batch, heads, source_length, target_length)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                # hidden_states = F.scaled_dot_product_attention(
                #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                # )
                hidden_states, scores = scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                if attn_name in patcher_self:
                    patcher_self[attn_name].append(scores)
                else:
                    patcher_self[attn_name] = [scores]

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states
            return new_call

        def _patch_block(block: torch.nn.Module, blocks_name: str, block_i: int) -> None:
            if hasattr(block, "attentions"):
                for xformer_i, xformer in enumerate(block.attentions):
                    for xformer_block_i, xformer_block in enumerate(xformer.transformer_blocks):
                        if patch_cross:
                            cross_attn = xformer_block.attn2
                            cross_attn.forward = get_attn_call(
                                cross_attn,
                                f"{blocks_name}.{block_i}.attentions.{xformer_i}.transformer_blocks.{xformer_block_i}.attn2"
                            )
                        if patch_self:
                            self_attn = xformer_block.attn1
                            self_attn.forward = get_attn_call(
                                self_attn,
                                f"{blocks_name}.{block_i}.attentions.{xformer_i}.transformer_blocks.{xformer_block_i}.attn1"
                            )

        for blocks_name in ["down_blocks", "up_blocks"]:
            blocks = getattr(unet, blocks_name)
            for block_i, block in enumerate(blocks):
                _patch_block(block, blocks_name, block_i)

        _patch_block(unet.mid_block, "mid_block", 0)
        setattr(unet, "attn_patched", True)

    def patch_call(patcher_self, pipeline: Any):

        def new_prepare_latents(
            image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True, sample_mode="sample"
        ):
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )

            latents_mean = latents_std = None
            if hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None:
                latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1)
            if hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None:
                latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1)

            # Offload text encoder if `enable_model_cpu_offload` was enabled
            if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
                pipeline.text_encoder_2.to("cpu")
                torch.cuda.empty_cache()

            image = image.to(device=device, dtype=dtype)

            batch_size = batch_size * num_images_per_prompt

            if image.shape[1] == 4:
                init_latents = image

            else:
                # make sure the VAE is in float32 mode, as it overflows in float16
                if pipeline.vae.config.force_upcast:
                    image = image.float()
                    pipeline.vae.to(dtype=torch.float32)

                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                elif isinstance(generator, list):
                    if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                        image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                    elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                        raise ValueError(
                            f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                        )

                    init_latents = [
                        retrieve_latents(pipeline.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode=sample_mode)
                        for i in range(batch_size)
                    ]
                    init_latents = torch.cat(init_latents, dim=0)
                else:
                    init_latents = retrieve_latents(pipeline.vae.encode(image), generator=generator, sample_mode=sample_mode)

                if pipeline.vae.config.force_upcast:
                    pipeline.vae.to(dtype)

                init_latents = init_latents.to(dtype)
                if latents_mean is not None and latents_std is not None:
                    latents_mean = latents_mean.to(device=device, dtype=dtype)
                    latents_std = latents_std.to(device=device, dtype=dtype)
                    init_latents = (init_latents - latents_mean) * pipeline.vae.config.scaling_factor / latents_std
                else:
                    init_latents = pipeline.vae.config.scaling_factor * init_latents

            if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // init_latents.shape[0]
                init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                init_latents = torch.cat([init_latents], dim=0)

            if add_noise:
                shape = init_latents.shape
                noise = patched_randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                # get latents
                init_latents = pipeline.scheduler.add_noise(init_latents, noise, timestep)

            latents = init_latents

            return latents

        @torch.no_grad()
        def new_pipe_call_img2img(
            prompt_orig: str,
            prompt_edit: str,
            num_repeat: int,
            sample_mode: str,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: PipelineImageInput = None,
            strength: float = 0.3,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            denoising_start: Optional[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):
            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )

            if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
                callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

            prompt = [prompt_orig, prompt_edit] * num_repeat

            # 1. Check inputs. Raise error if not correct
            pipeline.check_inputs(
                prompt,
                prompt_2,
                strength,
                num_inference_steps,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                callback_on_step_end_tensor_inputs,
            )

            pipeline._guidance_scale = guidance_scale
            pipeline._guidance_rescale = guidance_rescale
            pipeline._clip_skip = clip_skip
            pipeline._cross_attention_kwargs = cross_attention_kwargs
            pipeline._denoising_end = denoising_end
            pipeline._denoising_start = denoising_start
            pipeline._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = pipeline._execution_device

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                pipeline.cross_attention_kwargs.get("scale", None) if pipeline.cross_attention_kwargs is not None else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=pipeline.clip_skip,
            )

            # 4. Preprocess image
            image = pipeline.image_processor.preprocess(image)

            # 5. Prepare timesteps
            def denoising_value_valid(dnv):
                return isinstance(dnv, float) and 0 < dnv < 1

            timesteps, num_inference_steps = retrieve_timesteps(
                pipeline.scheduler, num_inference_steps, device, timesteps, sigmas
            )
            timesteps, num_inference_steps = pipeline.get_timesteps(
                num_inference_steps,
                strength,
                device,
                denoising_start=pipeline.denoising_start if denoising_value_valid(pipeline.denoising_start) else None,
            )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

            add_noise = True if pipeline.denoising_start is None else False

            # 6. Prepare latent variables
            if latents is None:
                latents = pipeline.prepare_latents_modified(
                    image,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    add_noise,
                    sample_mode
                )
            # 7. Prepare extra step kwargs.
            # extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

            height, width = latents.shape[-2:]
            height = height * pipeline.vae_scale_factor
            width = width * pipeline.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 8. Prepare added time ids & embeddings
            if negative_original_size is None:
                negative_original_size = original_size
            if negative_target_size is None:
                negative_target_size = target_size

            add_text_embeds = pooled_prompt_embeds
            if pipeline.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

            add_time_ids, add_neg_time_ids = pipeline._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                aesthetic_score,
                negative_aesthetic_score,
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

            if pipeline.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device)

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = pipeline.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    pipeline.do_classifier_free_guidance,
                )

            # 9. Denoising loop
            # num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)

            # 9.1 Apply denoising_end
            if (
                pipeline.denoising_end is not None
                and pipeline.denoising_start is not None
                and denoising_value_valid(pipeline.denoising_end)
                and denoising_value_valid(pipeline.denoising_start)
                and pipeline.denoising_start >= pipeline.denoising_end
            ):
                raise ValueError(
                    f"`denoising_start`: {pipeline.denoising_start} cannot be larger than or equal to `denoising_end`: "
                    + f" {pipeline.denoising_end} when using type float."
                )
            elif pipeline.denoising_end is not None and denoising_value_valid(pipeline.denoising_end):
                discrete_timestep_cutoff = int(
                    round(
                        pipeline.scheduler.config.num_train_timesteps
                        - (pipeline.denoising_end * pipeline.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]

            # 9.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if pipeline.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = pipeline.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            t = timesteps[0]

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents

            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipeline.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            patcher_self["noise"] = [noise_pred]

        pipeline.step_and_record = new_pipe_call_img2img
        pipeline.prepare_latents_modified = new_prepare_latents
        setattr(pipeline, "call_patched", True)
