'''from unet import UNetModel

unet = UNetModel(
    in_channels=3,
    model_channels=128,
    out_channels=6,
    num_res_blocks=3,
    attention_resolutions=tuple([4, 8]),
    dropout=0.0,
    channel_mult=(1, 2, 3, 4),
    num_classes=None,
    use_checkpoint=False,
    num_heads=4,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
)'''

'''
attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
'''

'''
if image_size == 256:
    channel_mult = (1, 1, 2, 2, 4, 4)
elif image_size == 64:
    channel_mult = (1, 2, 3, 4)
elif image_size == 32:
    channel_mult = (1, 2, 2, 2)
else:
    raise ValueError(f"unsupported image size: {image_size}")
'''

'''
model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
'''

'''
def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )
'''

'''
defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=100,  # 10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
'''
