dataset_type = 'IMDBDataset'
data_root = 'data/imdb/'
# location where the Glove vectors are cached (or will be downloaded)
vector_cache = 'data/vector_cache/'

data = dict(
    data_loader=dict(batch_size=8, num_workers=0),
    estimation=dict(
        type=dataset_type,
        root=data_root,
        vector_cache=vector_cache,
        split='train',
        select_cls='pos'),
    attribution=dict(
        type=dataset_type,
        root=data_root,
        vector_cache=vector_cache,
        split='train',
        select_cls='pos'))
