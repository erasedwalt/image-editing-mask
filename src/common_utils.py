import difflib
import inspect
import typing as tp
from typing import List, Optional, Tuple, Union

import PIL
import torch
from diffusers.utils.torch_utils import randn_tensor


def binarize(fm: torch.Tensor, threshold: float) -> torch.ByteTensor:
    return (fm > threshold).byte()


def rescale(tensor: torch.Tensor) -> torch.Tensor:
    min_ = tensor.min()
    max_ = tensor.max()
    return (tensor - min_) / (max_ - min_ + 1e-7)


def patched_randn_tensor(shape: Tuple | List, *args, **kwargs) -> torch.Tensor:
    bsz = shape[0]
    assert bsz % 2 == 0
    new_shape = (bsz // 2, *shape[1:])
    noise = randn_tensor(new_shape, *args, **kwargs)
    return noise.repeat(repeats=[2, 1, 1, 1])


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_timesteps(scheduler, num_inference_steps, strength, device, denoising_start=None):
    # get the original timestep using init_timestep
    if denoising_start is None:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start * scheduler.order)

        return timesteps, num_inference_steps - t_start

    else:
        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        discrete_timestep_cutoff = int(
            round(
                scheduler.config.num_train_timesteps
                - (denoising_start * scheduler.config.num_train_timesteps)
            )
        )

        num_inference_steps = (scheduler.timesteps < discrete_timestep_cutoff).sum().item()
        if scheduler.order == 2 and num_inference_steps % 2 == 0:
            # if the scheduler is a 2nd order scheduler we might have to do +1
            # because `num_inference_steps` might be even given that every timestep
            # (except the highest one) is duplicated. If `num_inference_steps` is even it would
            # mean that we cut the timesteps in the middle of the denoising step
            # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
            # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
            num_inference_steps = num_inference_steps + 1

        # because t_n+1 >= t_n, we slice the timesteps starting from the end
        t_start = len(scheduler.timesteps) - num_inference_steps
        timesteps = scheduler.timesteps[t_start:]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start)
        return timesteps, num_inference_steps


def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def get_add_time_ids(
        unet,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim,
    ):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    )
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError("Don't match.")

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)

    return add_time_ids


def encode_prompt(
        prompt,
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        clip_skip,
        device,
        prompt_2 = None,
        num_images_per_prompt = 1
    ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # batch_size = len(prompt)

    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]


    prompt_2 = prompt_2 or prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    # textual inversion: process multi-vector tokens if necessary
    prompt_embeds_list = []
    prompts = [prompt, prompt_2]
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        # if isinstance(self, TextualInversionLoaderMixin):
        #     prompt = self.maybe_convert_prompt(prompt, tokenizer)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    if text_encoder_2 is not None:
        prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )
    negative_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


@torch.inference_mode()
def prepare_latents(
    vae, scheduler, image, timestep, batch_size, num_images_per_prompt, dtype, device, sample_mode, generator=None, add_noise=True,
    randn_func=randn_tensor
):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        # make sure the VAE is in float32 mode, as it overflows in float16
        if vae.config.force_upcast:
            image = image.float()
            vae.to(dtype=torch.float32)

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
                retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i], sample_mode=sample_mode)
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode=sample_mode)

        if vae.config.force_upcast:
            vae.to(dtype)

        init_latents = init_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=device, dtype=dtype)
            latents_std = latents_std.to(device=device, dtype=dtype)
            init_latents = (init_latents - latents_mean) * vae.config.scaling_factor / latents_std
        else:
            init_latents = vae.config.scaling_factor * init_latents

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
        noise = randn_func(shape, generator=generator, device=device, dtype=dtype)
        # get latents
        init_latents = scheduler.add_noise(init_latents, noise, timestep)

    latents = init_latents

    return latents


def infer_unet(
        unet,
        latent_model_input,
        t,
        prompt_embeds,
        added_cond_kwargs,
):
    noise_pred = unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    return noise_pred


def high_level_get_timesteps(
        pipe,
        num_steps,
        batch_size,
        num_images_per_prompt,
        strength,
        denoising_start,
):
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_steps, pipe.device, timesteps=None, sigmas=None
    )
    timesteps, num_inference_steps = get_timesteps(
        pipe.scheduler,
        num_inference_steps,
        strength,
        pipe.device,
        denoising_start=denoising_start,
    )
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    add_noise = True if denoising_start is None else False

    return timesteps, latent_timestep, num_inference_steps, latent_timestep, add_noise


@torch.inference_mode()
def encode_prompt_with_shift(
        pipe,
        prompt,
        num_images_per_prompt,
        token_ids_to_shift,
        W_1,
        W_2,
        direction: tuple[tp.Literal["add_ones", "set_ones", "set_zeros", "k"], dict]
):
    (
        prompt_embeds_1,
        _,
        pooled_prompt_embeds_1,
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

    W_1_n = W_1.clone().detach()
    W_2_n = W_2.clone().detach()

    if direction[0] == "add_ones":
        eps = direction[1]["eps"]
        assert isinstance(eps, float)
        W_1_n[token_ids_to_shift] += eps * torch.nn.functional.normalize(torch.ones(W_1.shape[1]), p=2, dim=-1)[None] \
            .repeat(len(token_ids_to_shift), 1).half().to(pipe.device)
        W_2_n[token_ids_to_shift] += eps * torch.nn.functional.normalize(torch.ones(W_2.shape[1]), p=2, dim=-1)[None] \
            .repeat(len(token_ids_to_shift), 1).half().to(pipe.device)

    elif direction[0] == "set_ones":
        W_1_n[token_ids_to_shift] = torch.nn.functional.normalize(torch.ones(W_1.shape[1]), p=2, dim=-1)[None] \
            .repeat(len(token_ids_to_shift), 1).half().to(pipe.device)
        W_2_n[token_ids_to_shift] = torch.nn.functional.normalize(torch.ones(W_2.shape[1]), p=2, dim=-1)[None] \
            .repeat(len(token_ids_to_shift), 1).half().to(pipe.device)

    elif direction[0] == "set_zeros":
        W_1_n[token_ids_to_shift] = 0.
        W_2_n[token_ids_to_shift] = 0.

    elif direction[0] == "k":
        k = direction[1]["k"]
        assert isinstance(k, float)
        W_1_n[token_ids_to_shift] *= k
        W_2_n[token_ids_to_shift] *= k

    else:
        raise NotImplementedError()

    pipe.text_encoder.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(W_1_n.detach()).requires_grad_(True)
    pipe.text_encoder_2.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(W_2_n.detach()).requires_grad_(True)

    (
        prompt_embeds_2,
        _,
        pooled_prompt_embeds_2,
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

    pipe.text_encoder.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(W_1.detach()).requires_grad_(True)
    pipe.text_encoder_2.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(W_2.detach()).requires_grad_(True)

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2])
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2])

    return prompt_embeds, pooled_prompt_embeds


def prepare_inputs(
        pipe,
        batch_size,
        num_images_per_prompt,
        prompt_embeds,
        pooled_prompt_embeds,
        original_size,
        target_size,
):
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = get_add_time_ids(
        unet=pipe.unet,
        original_size=original_size,
        crops_coords_top_left=(0, 0),
        target_size=target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim
    )
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    prompt_embeds = prompt_embeds.to(pipe.device)
    add_text_embeds = add_text_embeds.to(pipe.device)
    add_time_ids = add_time_ids.to(pipe.device)

    return prompt_embeds, add_text_embeds, add_time_ids


def fix_timesteps_for_denoising_start_end(
        pipe,
        denoising_start,
        denoising_end,
        timesteps,
        num_inference_steps
):
    if (
        denoising_end is not None
        and denoising_start is not None
        and denoising_start >= denoising_end
    ):
        raise ValueError(
            f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
            + f" {denoising_end} when using type float."
        )
    elif denoising_end is not None:
        discrete_timestep_cutoff = int(
            round(
                pipe.scheduler.config.num_train_timesteps
                - (denoising_end * pipe.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    return timesteps, num_inference_steps


def find_needed_code_idxs(codes: list, target: str) -> list[int]:
    indices = []
    for i, code in enumerate(codes):
        if code[0] == target:
            indices.append(i)
    return indices


def get_slices_from_source(codes: list, indices: list[int]) -> list[slice]:
    changed_slices = []
    for code_idx in indices:
        code = codes[code_idx]
        slice_ = slice(code[1], code[2])
        changed_slices.append( slice_ )
    return changed_slices


def get_slices_from_target(codes: list, indices: list[int]) -> list[slice]:
    changed_slices = []
    for code_idx in indices:
        code = codes[code_idx]
        slice_ = slice(code[3], code[4])
        changed_slices.append( slice_ )
    return changed_slices


def get_token_ids_from_slices(tokens: list[int], slices: list[slice]) -> list[int]:
    token_ids = []
    for slice_ in slices:
        token_ids.extend(tokens[slice_])
    return token_ids


def handle_prompt_diffs(tokenizer, source: str, target: str) -> tuple[str, list[slice], list[int], list]:
    st, tt = tokenizer([source, target])["input_ids"]
    codes = difflib.SequenceMatcher(None, st, tt).get_opcodes()
    actions = set(x[0] for x in codes)

    changed_slices = []
    token_ids_to_shift = []
    use_source = False
    # use_target = False

    if "delete" in actions:
        code_indices = find_needed_code_idxs(codes, "delete")
        changed_slices.extend( get_slices_from_source(codes, code_indices) )
        token_ids_to_shift.extend( get_token_ids_from_slices(st, changed_slices) )
        use_source = True

    if "replace" in actions:
        code_indices = find_needed_code_idxs(codes, "replace")
        changed_slices.extend( get_slices_from_source(codes, code_indices) )
        token_ids_to_shift.extend( get_token_ids_from_slices(st, changed_slices) )
        use_source = True

    if "insert" in actions:
        code_indices = find_needed_code_idxs(codes, "insert")
        changed_slices.extend( get_slices_from_target(codes, code_indices) )
        token_ids_to_shift.extend( get_token_ids_from_slices(tt, changed_slices) )

    if use_source:
        prompt = source
    else:
        prompt = target

    return prompt, changed_slices, token_ids_to_shift, codes
