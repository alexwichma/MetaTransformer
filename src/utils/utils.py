"""
Miscellaneous utility functions
"""


from __future__ import absolute_import, division, print_function

import random
import numpy as np
import re
import torch
import os

from utils.taxonomy_utils import str_to_rank


def file_line_count(file_path):
    with open(file_path, "r") as file_handle:
        return sum(1 for _ in file_handle)


def load_class_weights(path, cfg):
    weight_dict = np.load(path).item()
    ranks = [str_to_rank(rank) for rank in cfg.multi_level_cls.ranks.split(",")]
    weightings = [weight_dict[rank] for rank in ranks]
    return weightings


def parse_class_mapping(file_path):
    mapping = {}
    with open(file_path, "r") as file_handle:
        lines = file_handle.readlines()[1:]
        for line in lines:
            split_line = line.lstrip().rstrip().split("\t")
            idx = int(split_line[0])
            class_name = split_line[1]
            mapping[idx] = class_name
    return mapping


def round_dataframe(pd_df, decimals=4):
    return pd_df.round(decimals)


def camel_case_to_snake_case(str):
    str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
    str = re.sub('(.)([0-9]+)', r'\1_\2', str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str).lower()


def snake_case_to_camel_case(str):
    str = re.sub('^([a-z])', lambda m: r'{}'.format(m.group(1).upper()), str)
    str = re.sub('_([a-z])', lambda m: r'{}'.format(m.group(1).upper()), str)
    return re.sub('_([0-9])', r'\1', str)


def create_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


SPECIAL_TOKENS = ['<pad>', '<unk>', '<cls>']
SPECIAL_TOKENS_2_INDEX = {'<pad>': 0, '<unk>': 1, '<cls>': 2}


def load_vocabulary(vocab_path):
    special_tokens = SPECIAL_TOKENS
    vocabulary = { s_token : count for count, s_token in enumerate(special_tokens) }
    index = len(special_tokens)
    with open(vocab_path, "r") as f_handle:
        for token in f_handle:
            token = token.strip()
            if token != "":
                vocabulary[token] = index
                index += 1
    vocab_len = index + 1
    return vocabulary, vocab_len


def regex_folder_match(path, regex):
    res = []
    for file in os.listdir(path):
        if re.match(regex, file):
            res.append(file)
    return res


# fix seeds for all rng's which are possibly used
def fix_random_seeds():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)