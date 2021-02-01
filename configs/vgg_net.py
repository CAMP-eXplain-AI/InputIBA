_base_ = ['_base_/imagenet.py']

model = dict(
    iba=dict(
        beta=20),
    net=dict(
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