dataset_type = 'ImageFolder'
data_root = 'data/derma_dataset/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_size = 300

train_pipeline = [
    dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensorV2')]

test_pipeline = [
    dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensorV2')]

data = dict(
    data_loader=dict(
        batch_size=1,
        shuffle=True,
        num_workers=0),
    estimation=dict(
        type=dataset_type,
        img_root=data_root + 'train/',
        pipeline=train_pipeline,
        valid_formats=['jpg', ]),
    attribution=dict(
        type=dataset_type,
        img_root=data_root + 'attribution/',
        pipeline=test_pipeline,
        valid_formats=['jpg', ])
)
