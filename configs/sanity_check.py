_base_ = ['./vgg_imagenet.py']


sanity_check = dict(
    perturb_layers=[
        'classifier.6',
        'classifier.0',
        'features.28',
        'features.24',
        'features.21',
        'features.17',
        'features.12',
        'features.7'],
    check='input_iba',
    verbose=False,
)
