import os.path as osp

import io
import torch
from collections import Counter
from torch.utils.data import IterableDataset
from torchtext.data.datasets_utils import (_add_docstring_header,
                                           _RawTextIterableDataset,
                                           _wrap_split_argument)
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import GloVe, Vocab

from .base import BaseDataset
from .builder import DATASETS

NUM_LINES = {'train': 25000, 'test': 25000}
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


@DATASETS.register_module()
class IMDBDataset(BaseDataset, IterableDataset):

    cls_to_ind = {'pos': 1, 'neg': 0}

    def __init__(self, root, vector_cache, split='train', select_cls=None):
        super(IMDBDataset, self).__init__()
        self.ind_to_cls = {v: k for k, v in self.cls_to_ind.items()}

        if select_cls is not None:
            assert select_cls in self.cls_to_ind, \
                f"select_cls must be None or one of " \
                f"{list(self.cls_to_ind.keys())}, but got {select_cls}"
            select_cls = [select_cls]
        else:
            select_cls = list(self.cls_to_ind.keys())
        self.select_cls = select_cls

        # build vocabulary and tokenizer
        _imdb_dataset = self._imdb(root, split)
        vec = GloVe(name='6B', dim=100, cache=vector_cache)
        self.tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        for data in _imdb_dataset:
            counter.update(self.tokenizer(data['input']))

        self._imdb_dataset = self._imdb(root, split)
        self.vocab = Vocab(counter, max_size=25000)
        self.vocab.load_vectors(vec)

    def text_to_tensor(self, text):
        return [self.vocab[t] for t in self.tokenizer(text)]

    def __iter__(self):
        for sample in self._imdb_dataset:
            input_text = sample['input']
            target = sample['target']
            if target in self.select_cls:
                input_name = sample['input_name']

                input_tensor = torch.tensor(
                    self.text_to_tensor(input_text), dtype=torch.long)
                target = self.cls_to_ind[target]
                input_name = osp.splitext(osp.basename(input_name))[0]
                input_length = input_tensor.shape[0]

                yield {
                    'input': input_tensor,
                    'target': target,
                    'input_name': input_name,
                    'input_length': input_length,
                    'input_text': input_text
                }

    @staticmethod
    @_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
    @_wrap_split_argument(('train', 'test'))
    def _imdb(root, split):

        def generate_imdb_data(key, extracted_files):
            for fname in extracted_files:
                if 'urls' in fname:
                    continue
                elif key in fname and ('pos' in fname or 'neg' in fname):
                    with io.open(fname, encoding="utf8") as f:
                        label = 'pos' if 'pos' in fname else 'neg'
                        yield {
                            'input': f.read(),
                            'target': label,
                            'input_name': fname
                        }

        dataset_tar = download_from_url(
            URL, root=root, hash_value=MD5, hash_type='md5')
        extracted_files = extract_archive(dataset_tar)
        iterator = generate_imdb_data(split, extracted_files)
        return _RawTextIterableDataset('IMDB', NUM_LINES[split], iterator)

    def get_cls_to_ind(self):
        return self.cls_to_ind

    def get_ind_to_cls(self):
        return self.ind_to_cls
