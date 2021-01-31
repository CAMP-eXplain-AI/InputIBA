_base_ = ['_base_/imagenet.py']

model = dict(
    net=dict(
        generator_iters=10,
        image_ib_beta=10,
        image_ib_opt_steps=40)
)

train_cfg = dict(
    n_samples=1000,
    progbar=True)

test_cfg = dict(
    feat_mask=dict(
        upscale=True,
        show=False),
    img_mask=dict(show=False))