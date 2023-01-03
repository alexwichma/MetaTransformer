"""
Helper functions to construct the correct model based on the config.yaml file provided for training.
"""


from __future__ import absolute_import, division, print_function

import logging
from tokenizers import Tokenizer

from layers.input_layers import EmbeddingLayer, HashEmbeddingLayer, OneHotEmbedding, OneHotEncoding
from layers.output_layers import FcLayers, MultiLevelClassificationLayer
from models.EmbedLstmAttention import EmbedLstmAttention
from models.ClassificationTransformer import ClassificationTransformer
from utils.transforms import LabelTransform, PrependClassificationToken, Read2HashKmer, Read2LshKmer, Read2VocabKmer, ReadToOneHotEncoding, TrimRead, ReadToBpeEncoding
from utils.utils import SPECIAL_TOKENS


logger = logging.getLogger(__name__)


def get_input_module(str, cfg, vocab_dim):
    """Loads the correct input module based on the configuration"""
    embed_dim = cfg.model.embed_dim
    num_buckets = 2 ** cfg.hashing.num_buckets
    num_hash_functions = cfg.hashing.num_hash_functions
    sparse = cfg.mdl_common.sparse_embedding if cfg.mdl_common.sparse_embedding is not None else False

    if str == 'vocab':
        inp_mod = EmbeddingLayer(vocab_dim, embed_dim, sparse)
    elif str == 'hash':
        corrected_num_buckets = num_buckets + len(SPECIAL_TOKENS)
        inp_mod = EmbeddingLayer(corrected_num_buckets , embed_dim, sparse)
    elif str == 'lsh':
        corrected_num_buckets = num_buckets + len(SPECIAL_TOKENS)
        inp_mod = EmbeddingLayer(corrected_num_buckets, embed_dim, sparse)
    elif str == 'hash_embedding':
        inp_mod = HashEmbeddingLayer(vocab_dim, embed_dim, num_buckets, num_hash_functions, sparse)
    elif str == 'one_hot':
        inp_mod = OneHotEncoding()
    elif str == 'one_hot_embed':
        inp_mod = OneHotEmbedding(embed_dim, sparse)
    elif str == 'bpe':
        bpe_model = Tokenizer.from_file(cfg.paths.bpe_model_path)
        inp_mod = EmbeddingLayer(bpe_model.get_vocab_size(), embed_dim, sparse)
    else:
        raise Exception("No valid embedding type provided: {}".format(str))
    
    return inp_mod


def get_output_module(str, cfg):
    """Loads the correct output module based on the configuration"""
    if str == "embed_lstm_attention":
        dim_inp = 2 * cfg.model.lstm_dim * cfg.model.num_att_heads
        dim_hidden = cfg.model.dense_dim
        # single level cls
        if not cfg.multi_level_cls.use:
            return FcLayers([dim_inp, dim_hidden, cfg.mdl_common.num_classes])
        # multi level cls
        else:
            num_classes_str = cfg.multi_level_cls.num_classes
            num_classes_lst = [int(class_num) for class_num in num_classes_str.split(",")]
            return MultiLevelClassificationLayer([dim_inp, dim_hidden], num_classes_lst)
    elif str == "classification_transformer":
        dim_inp = cfg.model.embed_dim
        if not cfg.multi_level_cls.use:
            return FcLayers([dim_inp, cfg.mdl_common.num_classes])
        else:
            num_classes_str = cfg.multi_level_cls.num_classes
            num_classes_lst = [int(class_num) for class_num in num_classes_str.split(",")]
            return MultiLevelClassificationLayer([dim_inp], num_classes_lst)
    else:
        raise Exception("No valid model str provided")

             
def read_transforms_for_input_layer(str, cfg, vocab, train=True):
    """Loads the correct input transformations based on the configuration"""
    kmer_sz = cfg.mdl_common.kmer_size
    num_buckets = 2 ** cfg.hashing.num_buckets

    transformations = [TrimRead()] if train else []

    if str == 'vocab':
        transformations.append(Read2VocabKmer(vocab, kmer_sz))
    elif str == 'hash':
        transformations.append(Read2HashKmer(num_buckets, kmer_sz))
    elif str == 'lsh':
        lsh_l = (kmer_sz // 2) + 1
        transformations.append(Read2LshKmer(num_buckets, kmer_sz, lsh_l))
    elif str == 'hash_embedding':
        transformations.append(Read2VocabKmer(vocab, kmer_sz))
    elif str == 'one_hot':
        transformations.append(ReadToOneHotEncoding())
    elif str == 'one_hot_embed':
        transformations.append(ReadToOneHotEncoding())
    elif str == 'bpe':
        bpe_model = Tokenizer.from_file(cfg.paths.bpe_model_path)
        transformations.append(ReadToBpeEncoding(bpe_model))
    # aggregation mode for trasnformer models
    if "aggregation_mode" in cfg.model and cfg.model.aggregation_mode == "cls":
        transformations.append(PrependClassificationToken())
    
    return transformations


def get_label_transforms(class_indices, header_sep="|",labeled=True):
    """Loads the correct label transformation based on the configuration"""
    if isinstance(class_indices, str):
        idxs = [int(idx) for idx in class_indices.split(",")]
        return LabelTransform(seq_id_idxs=idxs, header_sep=header_sep, labeled=labeled)
    elif isinstance(class_indices, int):
        return LabelTransform(seq_id_idxs=class_indices, header_sep=header_sep, labeled=labeled)
    else:
        raise Exception("No valid label indices provided. Please provide either a single int (e.g. 5) or a list of ints (e.g. 1, 3, 5) in the config file")


def instantiate_model_by_str_name(str, cfg, vocab_dim):
    """Instantiates the correct model based on the configuration"""

    input_module = get_input_module(cfg.mdl_common.input_module, cfg, vocab_dim)
    output_module = get_output_module(str, cfg)
    
    if str == 'embed_lstm_attention':
        model = EmbedLstmAttention(input_module, 
                                   output_module, 
                                   cfg.model.embed_dim,
                                   cfg.model.lstm_dim,
                                   cfg.model.att_dim,
                                   cfg.model.num_att_heads)
    elif str == 'classification_transformer':
        model = ClassificationTransformer(input_module,
                                          output_module,
                                          cfg.model.embed_dim,
                                          cfg.model.num_att_heads,
                                          cfg.model.dim_ff,
                                          cfg.model.num_encoder_blocks,
                                          cfg.model.dropout,
                                          cfg.model.aggregation_mode,
                                          cfg.device_settings.split_gpus,
                                          cfg.model.activation,
                                          cfg.model.use_pad_mask)
    else:
        raise Exception('No valid model with name {} found'.format(str))
    
    return model
