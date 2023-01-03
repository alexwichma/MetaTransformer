"""
This module contains all transforms for the input data
"""

from __future__ import absolute_import, division, print_function

from logging import getLogger
from typing import List, Union
import xxhash
import sourmash
import numpy as np
import random
import pickle

from cython_transforms import seq_to_kmer_lsh 
from utils.utils import SPECIAL_TOKENS, SPECIAL_TOKENS_2_INDEX


translation_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                    "K": "N", "M": "N", "R": "N", "Y": "N", "S": "N",
                    "W": "N", "B": "N", "V": "N", "H": "N", "D": "N",
                    "X": "N"}

one_hot_encoding = {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3, "A": 4, "C": 5, "G": 6, "T": 7}
one_hot_encoding_len = len(one_hot_encoding)


class TrimRead(object):

    def __init__(self, trim_min=0, trim_max=75) -> None:
        super().__init__()
        self.trim_min = trim_min
        self.trim_max = trim_max

    def __call__(self, x):
        trim_len = random.randint(self.trim_min, self.trim_max)
        end_index = len(x) - trim_len
        return x[:end_index]


class ReadToOneHotEncoding(object):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x):
        x_len = len(x)
        x_rc = forward2reverse(x)
        canonical = x if x < x_rc else x_rc
        result_indices = np.zeros((x_len,), dtype=np.int32)
        for index, base in enumerate(canonical):
            if not base in ["A", "C", "G", "T"]:
                result_indices[index] = one_hot_encoding["<unk>"]
            else:
                result_indices[index] = one_hot_encoding[base]
        return result_indices


class ReadToBpeEncoding(object):

    def __init__(self, bpe_model, canonical=True) -> None:
        super().__init__()
        self.bpe_model = bpe_model
        self.canonical = canonical

    def __call__(self, x):
        x = make_canonical(x) if self.canonical else x
        return self.bpe_model.encode(x).ids


class Read2VocabKmer(object):

    def __init__(self, vocabulary, k) -> None:
        super().__init__()
        self.logger = getLogger(self.__class__.__name__)
        self.vocabulary = vocabulary
        self.k = k
        self.kmer2index_fn = lambda kmer: kmer_to_index_vocab(self.vocabulary, kmer)

    def __call__(self, x):
        # return seq_to_kmer_vocab(x, self.k, self.vocabulary)
        return seq_to_kmer(x, self.k, self.kmer2index_fn)


class PrependClassificationToken(object):

    def __init__(self, cls_token_index=SPECIAL_TOKENS_2_INDEX["<cls>"]):
        super().__init__()
        self.cls_token_index = cls_token_index

    def __call__(self, x):
        return np.insert(x, 0, self.cls_token_index, axis=0)


class Read2HashKmer(object):

    def __init__(self, num_buckets, k) -> None:
        super().__init__()
        self.logger = getLogger(self.__class__.__name__)
        self.num_buckets = num_buckets
        self.k = k
        # keep special tokens (<pad>, <unk>, <cls> intact by shifting all other hashes)
        self.hash_add = len(SPECIAL_TOKENS)
        self.kmer2index_fn = lambda kmer: kmer2index_h(self.num_buckets, kmer, SPECIAL_TOKENS_2_INDEX) + self.hash_add

    def __call__(self, x):
        return seq_to_kmer(x, self.k, self.kmer2index_fn)

class Read2LshKmer(object):

    def __init__(self, num_buckets, k, l, num_min_hashes=None, seed_max_val=1000000) -> None:
        super().__init__()
        self.logger = getLogger(self.__class__.__name__)
        self.num_buckets = num_buckets
        self.k = k
        self.l = l
        self.hash_add = len(SPECIAL_TOKENS)
        self.num_min_hashes = (self.k - self.l + 1) if num_min_hashes is None else num_min_hashes
        # user specified number of min-hashes should at least cover the amounts lmers used
        # in our case: number of min-hashes is exactly equal to the number of extracted lmers since we do LSH
        assert self.num_min_hashes >= (self.k - self.l + 1)
        self.seeds = np.random.randint(0, seed_max_val, self.num_min_hashes, dtype=np.uint32)
        self.unk_idx = SPECIAL_TOKENS_2_INDEX["<unk>"]

    def __call__(self, x):
        # we need a bytes representation since the underlying cython implementation operates on char* dtype
        # standard encoding is "utf-8" for all versions of python3
        x = x.encode()
        return seq_to_kmer_lsh(x, self.k, self.l, self.num_buckets, self.seeds, self.unk_idx, self.hash_add)


class LabelTransform(object):

    def __init__(self, seq_id_idxs: Union[List[int], int], header_sep="|", labeled=True) -> None:
        super().__init__()
        self.seq_id_idxs = seq_id_idxs
        self.header_sep = header_sep
        self.labeled = labeled

    def __call__(self, header_line, labeled=True):
        split = header_line.split(self.header_sep)
        if self.labeled:
            if isinstance(self.seq_id_idxs, int):
                return int(split[self.seq_id_idxs])
            elif isinstance(self.seq_id_idxs, list):
                return [int(split[idx]) for idx in self.seq_id_idxs]
            else:
                raise Exception("No valid value for for seq_id_idxs provided")
        else:
            if isinstance(self.seq_id_idxs, int):
                return 0
            elif isinstance(self.seq_id_idxs, list):
                return [0 for idx in self.seq_id_idxs]
            else:
                raise Exception("No valid value for for seq_id_idxs provided")

def forward2reverse(seq):
    letters = list(seq)
    letters = [translation_dict[base] for base in letters]
    return ''.join(letters)[::-1]


def make_canonical(seq):
    seq_r = forward2reverse(seq)
    return seq if seq < seq_r else seq_r


def kmer_to_index_vocab(vocabulary, kmer_f):
    if kmer_f in vocabulary:
        return vocabulary[kmer_f]
    elif (kmer_r := forward2reverse(kmer_f)) in vocabulary:
        return vocabulary[kmer_r]
    else:
        return vocabulary["<unk>"]


# @DEPRECATED, faster cython implementation is used instead
def kmer2index_lsh(l, mask, half_min_hash_len, k_mer, special_tokens_mapping):
    if "N" in k_mer:
        return special_tokens_mapping['<unk>']
    canonical_kmer = make_canonical(k_mer)
    hash = xxhash.xxh32()
    minhash = sourmash.MinHash(n=len(canonical_kmer), ksize=l)
    minhash.add_sequence(canonical_kmer, True)
    # hashes should be already sorted
    min_hashes = minhash.get_hashes()
    sketch = min_hashes[:half_min_hash_len]
    hash.update(pickle.dumps(tuple(sketch)))
    return hash.intdigest() & mask


def kmer2index_h(num_buckets, k_mer, special_tokens_mapping):
    if "N" in k_mer:
        return special_tokens_mapping['<unk>']
    canonical_kmer = make_canonical(k_mer)
    mask = num_buckets - 1
    return xxhash.xxh32(canonical_kmer).intdigest() & mask


def seq_to_kmer(seq, k, kmer2index_fn):
    num_kmers = len(seq) - k + 1
    index_seq = np.zeros(shape=(num_kmers), dtype=np.int32)
    for index in range(0, num_kmers):
        kmer_f = seq[index:index + k]
        idx = kmer2index_fn(kmer_f)
        index_seq[index] = idx
    return index_seq
