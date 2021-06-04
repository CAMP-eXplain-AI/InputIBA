import torch
from torch.utils.data import DataLoader
import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import mmcv
from argparse import ArgumentParser
from PIL import Image

import iba
from iba.models import build_classifiers

# define IMDB dataset to return ID
import torchtext
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
import io

URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

MD5 = '7c2ac02c03563afcf9b574c7e56c153a'

NUM_LINES = {
    'train': 25000,
    'test': 25000,
}

_PATH = 'aclImdb_v1.tar.gz'

DATASET_NAME = "IMDB"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_wrap_split_argument(('train', 'test'))
def IMDB(root, split, cls=None):
    def generate_imdb_data(key, extracted_files):
        for fname in extracted_files:
            if 'urls' in fname:
                continue
            elif key in fname:
                if cls is None and ('pos' in fname or 'neg' in fname):
                    with io.open(fname, encoding="utf8") as f:
                        label = 'pos' if 'pos' in fname else 'neg'
                        yield label, f.read(), fname
                elif cls == 'pos' and ('pos' in fname):
                    with io.open(fname, encoding="utf8") as f:
                        label = 'pos'
                        yield label, f.read(), fname
                elif cls == 'neg' and ('neg' in fname):
                    with io.open(fname, encoding="utf8") as f:
                        label = 'neg'
                        yield label, f.read(), fname

    dataset_tar = download_from_url(URL, root=root,
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    iterator = generate_imdb_data(split, extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], iterator)


cfg = mmcv.Config.fromfile('/content/drive/MyDrive/informationbottleneck_merge/configs/deep_lstm.py')
dev = torch.device('cuda:0')

# load the data
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from torch.nn.utils.rnn import pad_sequence
import lime
from lime.lime_text import LimeTextExplainer

vec = torchtext.vocab.GloVe(name='6B', dim=100)
tokenizer = get_tokenizer('basic_english')
vocab_iter = torchtext.datasets.IMDB(split='train')

counter = Counter()
for (label, line) in vocab_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, max_size=25000)
vocab.load_vectors(vec)

def text_pipeline_lime(text):
    return [vocab[token] for token in tokenizer(text)]


def model_pred(model):
    def pred(texts):
        text_list = []
        for text in texts:
            processed_text = torch.tensor(text_pipeline_lime(text), dtype=torch.int64)
            text_list.append(processed_text)
        padded_text_list = pad_sequence(text_list)
        result = torch.sigmoid(model(padded_text_list, torch.tensor([padded_text_list.shape[0]]).expand(padded_text_list.shape[1])))
        pos_result = 1-result
        label = torch.cat([result, pos_result], dim=1)
        return label.detach().numpy()
    return pred


# normalize saliency to be in [0.2, 1]
def normalize_saliency(exp):
    exp.as_list(label=0)
    saliency = [i[1] for i in exp.as_list(label=0)]
    saliency = torch.tensor(saliency)
    saliency -= saliency.min()
    saliency /= (saliency.max()/0.8)
    saliency += 0.2
    return saliency


def parse_args():
    parser = ArgumentParser('Generate attribution masks using LIME')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument('--num-samples',
                        type=int,
                        default=2000,
                        help='Number of samples to evaluate, 0 means checking all the samples')
    args = parser.parse_args()
    return args


# get masks
def generate_lime_masks(cfg=cfg,
                  work_dir=None,
                  num_samples=2000):
    tokenizer = get_tokenizer('basic_english')
    train_iter = IMDB(split='test', cls='pos')
    model = build_classifiers(cfg.attributor['classifier'])
    model = model.eval()
    explainer = LimeTextExplainer()
    for count, (label, text, filename) in tqdm(enumerate(train_iter)):
        if count >= num_samples:
            break
        filename = filename.split('/')[-1].split('.')[0]
        mask_file = os.path.join(work_dir, filename)
        if not os.path.isfile(mask_file + '.png'):
            num_label = 0 if 'neg' else 1
            exp = explainer.explain_instance(text, model_pred(model), labels=[num_label, ], num_samples=200, num_features=20)

            # normalize saliency
            saliency = normalize_saliency(exp)

            # generate input mask given lime result
            feature = [i[0] for i in exp.as_list(label=0)]
            tokenized_text = [token for token in tokenizer(text)]
            word_saliency = torch.zeros((len(tokenized_text),))
            for feat_index, feat in enumerate(feature):
                for index, word in enumerate(tokenized_text):
                    if feat.lower() == word:
                        word_saliency[index] = saliency[feat_index]
            mask = word_saliency.numpy()

            # save mask as png image
            mask = (mask * 255).astype(np.uint8)
            mask = np.resize(np.expand_dims(mask, 0), (50, mask.shape[0]))
            dir_name = osp.abspath(osp.dirname(mask_file))
            mmcv.mkdir_or_exist(dir_name)
            mask = Image.fromarray(mask, mode='L')
            mask.save(mask_file + '.png')


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    generate_lime_masks(cfg=cfg,
                  work_dir=args.work_dir,
                  num_samples=args.num_samples)


if __name__ == '__main__':
    main()
