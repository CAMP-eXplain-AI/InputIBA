dataset_type = 'CIFAR10'
data_root = 'data/cifar10/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_size = 32

train_pipeline = [
    dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensor')]

test_pipeline = [
    dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensor')]

data = dict(
    data_loader=dict(
        batch_size=1,
        shuffle=True,
        num_workers=0),
    train=dict(
        type=dataset_type,
        img_root=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_root=data_root + 'train/',
        pipeline=test_pipeline)
)