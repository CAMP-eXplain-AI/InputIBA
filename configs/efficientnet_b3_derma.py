_base_ = ['_base_/derma_dataset.py']

checkpoint_path = 'workdirs/ckpts/efficientnet_b4_derma.pt'

attributor = dict(
    type='VisionAttributor',
    layer='blocks.6',
    use_softmax=True,
    classifier=dict(
        source='timm',
        type='efficientnet_b4',
        num_classes=10,
        pretrained=False,
        checkpoint_path=checkpoint_path),
    feat_iba=dict(
        type='VisionFeatureIBA',
        input_or_output="output",
        active_neurons_threshold=0.01,
        initial_alpha=5.0),
    input_iba=dict(
        type='VisionInputIBA',
        initial_alpha=5.0,
        sigma=1.0),
    gan=dict(
        type='VisionWGAN',
        generator=dict(type='VisionGenerator'),
        discriminator=dict(type='VisionDiscriminator'))
)

estimation_cfg = dict(
    n_samples=1000,
    verbose=False,
)

attribution_cfg = dict(
    feat_iba=dict(
        batch_size=10,
        beta=20,
        log_every_steps=-1),
    gan=dict(
        dataset_size=200,
        sub_dataset_size=20,
        lr=0.00005,
        batch_size=32,
        weight_clip=0.01,
        epochs=20,
        critic_iter=5,
        verbose=False),
    input_iba=dict(
        beta=20.0,
        opt_steps=60,
        lr=1.0,
        batch_size=10,
        log_every_steps=-1),
    feat_mask=dict(
        upscale=True,
        show=False),
    input_mask=dict(
        show=False)
)