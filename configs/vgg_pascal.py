_base_ = ['_base_/pascal.py']

pretrained = 'workdirs/ckpts/vgg16_voc.pth'

attributor = dict(
    layer='features.17',
    use_softmax=True,
    classifier=dict(
        type='vgg16',
        num_classes=20,
        pretrained=pretrained),
    feat_iba=dict(
        input_or_output="output",
        active_neurons_threshold=0.01,
        initial_alpha=5.0),
    input_iba=dict(
        initial_alpha=5.0,
        sigma=1.0),
    gan=dict(
        generator=dict(type='VisionGenerator'),
        discriminator=dict(type='VisionDiscriminator'))
)

estimation_cfg = dict(
    n_samples=1000,
    progbar=True,
)

attribution_cfg = dict(
    feat_iba=dict(
        batch_size=10,
        beta=10.0),
    gan=dict(
        dataset_size=200,
        sub_dataset_size=20,
        lr=0.00005,
        batch_size=32,
        weight_clip=0.01,
        epochs=20,
        critic_iter=5),
    input_iba=dict(
        beta=10.0,
        opt_steps=60,
        lr=1.0,
        batch_size=10),
    feat_mask=dict(
        upscale=True,
        show=False),
    input_mask=dict(
        show=False)
)