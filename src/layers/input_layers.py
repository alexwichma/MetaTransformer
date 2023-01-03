from __future__ import absolute_import, division, print_function

from math import sqrt
from typing import List
import torch
from torch import nn
from torch.tensor import Tensor
from torch.utils.data import DataLoader
import random

from dataset.MetagenomicReadDataset import ProcessingMetagenomicReadDataset
from utils.utils import SPECIAL_TOKENS, SPECIAL_TOKENS_2_INDEX, load_vocabulary
from utils.transforms import TrimRead, Read2VocabKmer, one_hot_encoding_len, one_hot_encoding


class EmbeddingLayer(nn.Module):
    """Wrapper around a simple embedding layer"""

    def __init__(self, vocab_dim, embed_dim, sparse):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=SPECIAL_TOKENS_2_INDEX["<pad>"], sparse=sparse)

    def forward(self, x):
        # Dim(x) = (batch_size, seq_length)
        # Dim(out) = (batch_size, seq_length, embed_dim)
        return self.embedding(x)


class HashEmbeddingLayer(nn.Module):
    """Hash embedding layer as described in the publication by Georgiou et al."""

    def __init__(self, vocab_dim, embed_dim, num_buckets, num_hashes, sparse, preserve_special_tokens=True):
        super(HashEmbeddingLayer, self).__init__()
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.num_hashes = num_hashes
        self.index_correction = len(SPECIAL_TOKENS) if preserve_special_tokens else 0
        self.p = self._get_prime_larger_than(random.randint(vocab_dim, 2**32))
        self.m = num_buckets
        # one additional bucket is needed to account for shift in case that padding should stay at zero
        self.num_buckets = num_buckets + self.index_correction
        # a set could be used here since the values should be unique anyways, but set is unordered
        # and usually only a small amount of hash functions are needed, so the overhead for filling
        # those arrays is really small
        self.a_vals = []
        self.b_vals = []
        for _ in range(num_hashes):
            self._construct_hash()

        # stores the shared embedding vectors
        self.shared_embedding = nn.Embedding(self.num_buckets, self.embed_dim, padding_idx=0, sparse=sparse)
        # stores the weightings of the "num_hashes" hash functions for each possible token
        self.hash_weights= nn.Embedding(self.vocab_dim, self.num_hashes, padding_idx=0, sparse=sparse)

    def _is_prime(self, x):
        for n in range(2, 1 + int(sqrt(x))):
           if x % n == 0:
               return False
        return True

    def _get_prime_larger_than(self, x):
        candidate = x + 1
        while not self._is_prime(candidate):
            candidate += 1
        return candidate

    def _construct_hash(self):
        candidate_a = None
        candidate_b = None
        while candidate_a is None or candidate_a in self.a_vals:
            candidate_a = random.randint(1, self.p - 1)
        while candidate_b is None or candidate_b in self.b_vals:
            candidate_b = random.randint(0, self.p - 1)
        self.a_vals.append(candidate_a)
        self.b_vals.append(candidate_b)

    def _calc_hash(self, a, b, x: Tensor) -> Tensor:
        return (((a * x + b) % self.p) % self.m) + self.index_correction

    def _apply_hash_family(self, x: Tensor) -> List[Tensor]:
        return [self._calc_hash(a, b, x) for a, b in zip(self.a_vals, self.b_vals)] 
        
    def forward(self, x):
        # dim x = (batch_sz, seq_len)
        hash_weights = torch.unsqueeze(self.hash_weights(x), dim=-1)  # dim = (batch_sz, seq_len, num_hashes, 1)
        hashes_list = self._apply_hash_family(x)  # dim = list [(batch_size, seq_len)] of length num_hashes
        hashes = torch.stack(hashes_list, dim=-1)  # dim = (batch_sz, seq_len, num_hashes)
        # preserve special indices (<pad>, <unk>, <cls> ...) by remapping them to their original values
        x_unsqueezed = torch.unsqueeze(x, -1)
        for _, s_index in SPECIAL_TOKENS_2_INDEX.items():
            hashes = hashes.masked_fill_(x_unsqueezed == s_index, s_index)
        unweighted_shared_embs = self.shared_embedding(hashes)  # dim = (batch_sz, seq_len, num_hashes, embed_dim)
        weighted_shared_embs = hash_weights * unweighted_shared_embs  # dim = (batch_sz, seq_len, num_hashes, embed_dim), works because of broadcast of "hash_weights"
        aggregated_shared_embs = torch.sum(weighted_shared_embs, dim=-2) # dim = (batch_sz, seq_len, embed_dim)
        return aggregated_shared_embs


class OneHotEncoding(nn.Module):
    """Simple one-hot-encoding"""

    def __init__(self):
        super(OneHotEncoding, self).__init__()

    def forward(self, x):
        return torch.nn.functional.one_hot(x, one_hot_encoding_len)


class OneHotEmbedding(nn.Module):
    """Wrapper around an embedding used for one-hot-encoding"""

    def __init__(self, embed_dim, sparse):
        super(OneHotEmbedding, self).__init__()
        self.embedding = nn.Embedding(one_hot_encoding_len, embed_dim, padding_idx=one_hot_encoding["<pad>"], sparse=sparse)
    
    def forward(self, x):
        return self.embedding(x)


if __name__ == "__main__":
    print("Loading vocab")
    vocabulary, vocab_len = load_vocabulary("../data/shared/vocabs/vocab_12mer.txt")
    print("Done. Vocab length: {}".format(vocab_len))
    transforms = [TrimRead(), Read2VocabKmer(vocabulary, 12, 150)]
    processing_ds = ProcessingMetagenomicReadDataset("../data/hgr_umgs/12mer_plain/train_genus", vocabulary, 12, transforms)
    dl = DataLoader(processing_ds, batch_size=2048, num_workers=1)
    print("Initializing hash embedding layer")
    hash_embedding = HashEmbeddingLayer(vocab_len, 300, 2**20, 2)
    print("Done")
    num_reads = 0
    for data, target in dl:
        print("Data shape: {}, Target shape: {}".format(data.shape, target.shape))
        num_reads += data.shape[0]
        print(hash_embedding(data).shape)
    print("Number of reads: {}".format(num_reads))
    