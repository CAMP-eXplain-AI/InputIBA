dataset_type = 'ImageNet'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_size = 224
train_pipeline = [
    dict(type='Resize', size=(img_size, img_size)),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)]

test_pipeline = [
    dict(type='Resize', size=(img_size, img_size)),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)]


data = dict(
    data_loader=dict(
        batch_size=1,
        shuffle=True,
        num_workers=0),
    train=dict(
        type=dataset_type,
        root=data_root + 'imagenet/train/',
        ind_to_cls_file=data_root + 'imagenet_class_index.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        root=data_root + 'imagenet/val/',
        ind_to_cls_file=data_root + 'imagenet_class_index.json',
        pipeline=test_pipeline))