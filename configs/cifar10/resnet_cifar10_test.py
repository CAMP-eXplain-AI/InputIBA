_base_ = ['./resnet_cifar10_train.py']

data_root = 'data/cifar10/'
data = dict(attribution=dict(img_root=data_root + 'test/'))
