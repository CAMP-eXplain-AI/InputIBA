_base_ = ['_base_/cifar10.py']

pretrained = 'workdirs/ckpts/small_vgg_cifar10.pth'

attributor = dict(
    layer='features.14',
    use_softmax=True,
    classifier=dict(
        source='custom',
        type='SmallVGG',
        num_classes=10,
        pretrained=pretrained),
    iba=dict(
        input_or_output="output",
        active_neurons_threshold=0.01,
        initial_alpha=5.0),
    img_iba=dict(
        initial_alpha=5.0,
        sigma=1.0,
    )
)

estimation_cfg = dict(
    n_samples=1000,
    progbar=True,
)

attribution_cfg = dict(
    iba=dict(
        batch_size=10,
        beta=20),
    gan=dict(
        dataset_size=200,
        sub_dataset_size=20,
        lr=0.00005,
        batch_size=32,
        weight_clip=0.01,
        epochs=20,
        critic_iter=5),
    img_iba=dict(
        beta=20.0,
        opt_steps=60,
        lr=1.0,
        batch_size=10),
    feat_mask=dict(
        upscale=True,
        show=False),
    img_mask=dict(
        show=False)
)