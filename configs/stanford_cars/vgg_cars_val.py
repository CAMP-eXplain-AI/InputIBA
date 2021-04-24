_base_ = ['./vgg_cars_train.py']
data_root = 'data/stanford_cars/'
data = dict(
    val=dict(
        img_root=data_root + 'validation/'
    )
)