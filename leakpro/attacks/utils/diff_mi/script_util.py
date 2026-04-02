"""Utilities for creating models and diffusion processes."""

from typing import Literal

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import EncoderUNetModel, UNetModel

NUM_CLASSES = 1000 + 1

def model_and_diffusion_defaults() -> dict:
    """Defaults for image training."""
    res = {
        "image_size": 64,
        "num_channels": 128,
        "num_res_blocks": 3,
        "num_classes": NUM_CLASSES,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "num_head_channels": -1,
        "attention_resolutions": "16,8",
        "channel_mult": "",
        "dropout": 0.0,
        "class_cond": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_fp16": False,
        "use_new_attention_order": False,
    }
    res.update(diffusion_defaults())
    return res

def model_defaults() -> dict:
    """Defaults for image training."""
    return {
        "image_size": 64,
        "num_channels": 128,
        "num_res_blocks": 3,
        "num_classes": NUM_CLASSES,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "num_head_channels": -1,
        "attention_resolutions": "16,8",
        "channel_mult": "",
        "dropout": 0.0,
        "class_cond": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_fp16": False,
        "use_new_attention_order": False,
    }

def diffusion_defaults() -> dict:
    """Defaults for diffusion training."""
    return {
        "learn_sigma": False,
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "timestep_respacing": "",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
    }

def create_model_and_diffusion(
    image_size: int,
    class_cond: bool,
    num_classes: int,
    learn_sigma: bool,
    num_channels: int,
    num_res_blocks: int,
    channel_mult: str,
    num_heads: int,
    num_head_channels: int,
    num_heads_upsample: int,
    attention_resolutions: str,
    dropout: float,
    diffusion_steps: int,
    noise_schedule: str,
    timestep_respacing: list[int] | str,
    use_kl: bool,
    predict_xstart: bool,
    rescale_timesteps: bool,
    rescale_learned_sigmas: bool,
    use_checkpoint: bool,
    use_scale_shift_norm: bool,
    resblock_updown: bool,
    use_fp16: bool,
    use_new_attention_order: bool,
    w: float | None = None,
) -> tuple[UNetModel, SpacedDiffusion]:
    """Create a model and diffusion process."""
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        diffusion_steps=diffusion_steps,
        w=w,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def create_model(
    image_size: int,
    num_channels: int,
    num_res_blocks: int,
    channel_mult: str = "",
    learn_sigma: bool = False,
    class_cond: bool = False,
    use_checkpoint: bool = False,
    attention_resolutions: str = "16",
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    num_classes: int = NUM_CLASSES,
) -> UNetModel:
    """Create a U-Net model."""
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

def create_classifier_and_diffusion(
    image_size: int,
    classifier_use_fp16: bool,
    classifier_width: int,
    classifier_depth: int,
    classifier_attention_resolutions: str,
    classifier_use_scale_shift_norm: bool,
    classifier_resblock_updown: bool,
    classifier_pool: str,
    learn_sigma: bool,
    diffusion_steps: int,
    noise_schedule: str,
    timestep_respacing: list[int] | str,
    use_kl: bool,
    predict_xstart: bool,
    rescale_timesteps: bool,
    rescale_learned_sigmas: bool,
) -> tuple[EncoderUNetModel, SpacedDiffusion]:
    """Create a classifier and diffusion process."""
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        diffusion_steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion

def create_classifier(
    image_size: int,
    classifier_use_fp16: bool,
    classifier_width: int,
    classifier_depth: int,
    classifier_attention_resolutions: str,
    classifier_use_scale_shift_norm: bool,
    classifier_resblock_updown: bool,
    classifier_pool: Literal["adaptive", "attention", "spatial", "spatial_v2"],
    out_channels: int = 1000,
) -> EncoderUNetModel:
    """Create a U-Net classifier model."""
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def create_gaussian_diffusion(
    *,
    diffusion_steps: int = 1000,
    w: float = 1.0,
    learn_sigma: bool =False,
    sigma_small: bool =False,
    noise_schedule: str = "linear",
    use_kl: bool =False,
    predict_xstart: bool =False,
    rescale_timesteps: bool =False,
    rescale_learned_sigmas: bool =False,
    timestep_respacing: list[int] | str = "",
) -> SpacedDiffusion:
    """Create a Gaussian diffusion process."""
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        w=w,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
