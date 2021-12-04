import mmcv
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm


def filter_samples(dataset, name_to_score_dict, threshold=0.6):
    valid_inds = []
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        score = name_to_score_dict[sample['input_name']]
        if score >= threshold:
            valid_inds.append(i)
    return valid_inds


def get_valid_set(dataset,
                  scores_file=None,
                  scores_threshold=0.6,
                  num_samples=0):
    valid_inds = np.arange(len(dataset))
    if scores_file is not None:
        scores_dict = mmcv.load(scores_file)
        name_to_score_dict = {k: v['pred'] for k, v in scores_dict.items()}
        valid_inds = filter_samples(
            dataset, name_to_score_dict, threshold=scores_threshold)

    if num_samples > 0:
        num_valid_samples = min(num_samples, len(valid_inds))
        valid_inds = np.random.choice(
            valid_inds, num_valid_samples, replace=False)
    print(f'Total samples: {len(valid_inds)}')
    valid_set = Subset(dataset, valid_inds)
    return valid_set
