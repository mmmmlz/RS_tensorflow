# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/15  15:39 
# @file   ：tools.py.PY
# @Tool   ：PyCharm
from feature_column import SparseFeat,VarLenSparseFeat,DenseFeat
from tensorflow.python.keras.layers import Input, Lambda,Embedding
from tensorflow.python.keras.regularizers import l2
import numpy as np
from collections import OrderedDict,defaultdict
import feature_column as fc_lib
from itertools import chain
from layers import Hash,NoMask
import tensorflow as tf
from tensorflow.python.keras import backend as K



def get_xy_fd_sdm(hash_flag=False):
    user_feature_columns = [SparseFeat('user', 3),
                            SparseFeat('gender', 2),
                            VarLenSparseFeat(SparseFeat('prefer_item', vocabulary_size=100, embedding_dim=8,
                                                        embedding_name='item'), maxlen=6,
                                             length_name="prefer_sess_length"),
                            VarLenSparseFeat(SparseFeat('prefer_cate', vocabulary_size=100, embedding_dim=8,
                                                        embedding_name='cate'), maxlen=6,
                                             length_name="prefer_sess_length"),
                            VarLenSparseFeat(SparseFeat('short_item', vocabulary_size=100, embedding_dim=8,
                                                        embedding_name='item'), maxlen=4,
                                             length_name="short_sess_length"),
                            VarLenSparseFeat(SparseFeat('short_cate', vocabulary_size=100, embedding_dim=8,
                                                        embedding_name='cate'), maxlen=4,
                                             length_name="short_sess_length"),
                            ]
    #
    item_feature_columns = [SparseFeat('item', 100, embedding_dim=8, ),
                            SparseFeat("cate",100,embedding_dim=8)
                            ]

    uid = np.array([0, 1, 2, 1])
    ugender = np.array([0, 1, 0, 1])
    iid = np.array([1, 2, 3, 1])  # 0 is mask value
    i_cate = np.array([1,2,3,1])

    prefer_iid = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0], [1, 2, 3, 3, 0, 0], [1, 2, 4, 0, 0, 0]])
    prefer_cate = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0], [1, 2, 3, 3, 0, 0], [1, 2, 4, 0, 0, 0]])
    short_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [3, 0, 0, 0]])
    short_cate = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [3, 0, 0, 0]])
    prefer_len = np.array([6, 5, 4, 3])
    short_len = np.array([3, 3, 2, 1])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid,'cate':i_cate, 'prefer_item': prefer_iid, "prefer_cate": prefer_cate,
                    'short_item': short_iid, 'short_cate': short_cate, 'prefer_sess_length': prefer_len,
                    'short_sess_length': short_len}

    # feature_names = get_feature_names(feature_columns)
    x = feature_dict
    y = np.array([1, 1, 1, 0])
    history_feature_list = ['item', 'cate']

    return x, y, user_feature_columns, item_feature_columns, history_feature_list


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features

def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):


    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    #print("122",sparse_feature_columns)
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        #print(feat.name, feat.embedding_name)

        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            #print(feat.name,feat.embedding_name)
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.embedding_name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb

    return sparse_embedding


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                # 如果是字符串形式的类别特征，则需要进行编码
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list), vocabulary_path=fc.vocabulary_path)(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict

def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True, vocabulary_path=fc.vocabulary_path)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list

def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)
def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)
def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)
