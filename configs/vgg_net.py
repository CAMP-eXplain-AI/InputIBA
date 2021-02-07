_base_ = ['_base_/imagenet.py']

model = dict(
    classifier=dict(
        type='vgg16',
        pretrained=True),
    iba=dict(
        layer='features.17',
        beta=20),
    net=dict(
        position="features[17]",
        epochs=10,
        image_ib_beta=20,
        image_ib_opt_steps=60)
)

train_cfg = dict(
    n_samples=1000,
    progbar=True)

test_cfg = dict(
    feat_mask=dict(
        upscale=True,
        show=False),
    img_mask=dict(show=False))