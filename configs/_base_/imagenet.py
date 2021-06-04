dataset_type = 'ImageNet'
data_root = 'data/imagenet/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_size = 224
estimation_pipeline = [
    dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensor')]

attribution_pipeline = [
    dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensor')]


data = dict(
    data_loader=dict(
        batch_size=1,
        shuffle=True,
        num_workers=0),
    estimation=dict(
        type=dataset_type,
        img_root=data_root + 'images/estimation/',
        ind_to_cls_file=data_root + 'imagenet_class_index.json',
        pipeline=estimation_pipeline,
        with_bbox=False),
    attribution=dict(
        type=dataset_type,
        img_root=data_root + 'images/attribution/',
        annot_root=data_root + 'annotations/attribution/',
        ind_to_cls_file=data_root + 'imagenet_class_index.json',
        pipeline=attribution_pipeline,
        with_bbox=True))
