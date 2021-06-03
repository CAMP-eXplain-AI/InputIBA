_base_ = ['_base_/imagenet.py']

pretrained = 'ckpts/deep_lstm.pth'

attributor = dict(
    type='NLPAttributor',
    layer='rnn_4',
    use_softmax=True,
    # Recurrent model need to be in train model to allow for back propagation
    eval_classifier=False,
    classifier=dict(
        source='custom',
        type='DeepLSTM',
        pretrained=pretrained),
    feat_iba=dict(
        type='NLPFeatureIBA',
        input_or_output="output",
        active_neurons_threshold=0.01,
        initial_alpha=5.0),
    input_iba=dict(
        type='NLPInputIBA',
        initial_alpha=5.0,
        sigma=0.0),
    gan=dict(
        type='NLPWGAN',
        generator=dict(type='NLPGenerator'),
        discriminator=dict(type='NLPDiscriminator'))
)

estimation_cfg = dict(
    n_samples=1000,
    verbose=False,
)

attribution_cfg = dict(
    feat_iba=dict(
        batch_size=10,
        beta=0.5,
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
        beta=0.01,
        opt_steps=20,
        lr=0.5,
        batch_size=10,
        log_every_steps=-1),
    feat_mask=dict(
        upscale=True,
        show=False),
    input_mask=dict(
        show=False)
)
