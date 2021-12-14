# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/9  16:16 
# @file   ：tools.py
# @Tool   ：PyCharm
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.regularizers import l2
from feature_column import SparseFeat,VarLenSparseFeat,DenseFeat
import numpy as np
from collections import OrderedDict
from tensorflow.python.keras.layers import Input, Lambda
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from itertools import chain
from collections import defaultdict
from layer import NoMask
from tensorflow.python.keras import backend as K

ALL_FEATURES=['user', 'gender', 'item',"price",'hist_item',"hist_len","i_cate_id","i_brand_id"]
ITEM_FEATURES=["item","i_cate_id","i_brand_id"]
USER_FEATURES=["user","gender","price","hist_item","hist_cate","hist_brand","hist_len"]
SPARSE_FEATURES = ["user","gender"]
DENSE_FEATURES =["price"]
# 将我们构造的输入假数据变成tensor 用于模型初始化
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

# 制作一些测试数据，需要x,y 真实的用于训练的数据，user_feature_columns，item_feature_columns 假数据 用于模型初始化
def make_test_data(hash_flag=False):
    user_feature_columns = [SparseFeat('user', 3), SparseFeat('gender', 2),
        VarLenSparseFeat(SparseFeat('hist_item', vocabulary_size=3 + 1, embedding_dim=4, embedding_name='item'), maxlen=4,length_name="hist_len"),
        VarLenSparseFeat(SparseFeat('hist_i_cate_id', vocabulary_size=3 + 1, embedding_dim=4, embedding_name='item'), maxlen=4,length_name="hist_len"),
        VarLenSparseFeat(SparseFeat('hist_i_brand_id', vocabulary_size=3 + 1, embedding_dim=4, embedding_name='item'), maxlen=4,length_name="hist_len"),
        DenseFeat("price")]

    item_feature_columns = [SparseFeat('item', 3 + 1, embedding_dim=4, ),
                            SparseFeat('i_cate_id',4,embedding_dim=4),
                            SparseFeat("i_brand_id",4,embedding_dim=4)]

    uid = np.array([0, 1, 2, 1])
    ugender = np.array([0, 1, 0, 1])

    iid = np.array([1, 2, 3, 1])  # 0 is mask value
    i_cate_id = np.array([2,3,1,0])
    i_brand_id = np.array([2,2,1,1])

    price = np.array([1.3, 2.1, 3.1, 3.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [3, 0, 0, 0]])
    hist_cate = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [3, 0, 0, 0]])
    hist_brand = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [3, 0, 0, 0]])


    hist_len = np.array([3, 3, 2, 1])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, "price": price,
                    'hist_item': hist_iid, "hist_len": hist_len,
                    "i_cate_id": i_cate_id, "i_brand_id": i_brand_id,
                    "hist_i_cate_id": hist_cate,"hist_i_brand_id":hist_brand

                    }

    # feature_names = get_feature_names(feature_columns)
    x = feature_dict
    y = np.array([1, 1, 1, 1])
    return x, y, user_feature_columns, item_feature_columns


def pooling(his_list,flag="1"):
    if len(his_list) ==1:
        return his_list[0]
    tmp = list(map(lambda x: tf.expand_dims(x,axis=-1),his_list))
    tmp = tf.keras.layers.concatenate(tmp,axis=-1)
    print("tmp",tmp)
    tf.print(tmp)
    if flag == "2":
        tmp = tf.reshape(tmp,[-1,4,3])
    res = tf.reduce_mean(tmp,axis=-1,name="3213"+flag,)
    return res

def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding





def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    '''
    :param sparse_embedding_dict: 初始化的embedding 字典
    :param sparse_input_dict:    输入的特征 字典 tensor
    :param sparse_feature_columns: 输入特征对应的信息 Feat
    :param return_feat_list:
    :param mask_feat_list:
    :param to_list:
    :return: 根据sparse_feature_columns里面的特征信息，在sparse_input_dict取出对应的输入tensor，在sparse_embedding_dict去查找对应的embedding
            如果特征在return_feat_list里面的话


    '''
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name

        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    print(group_embedding_dict)
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def get_dense_input(features, feature_columns):
    import feature_column as fc_lib
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list
def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)

def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        # Flatten [B,1,8]->[B,8]

        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        #print(dense_dnn_input)
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")

def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)
def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)


if __name__ == '__main__':
    x,y,user_feat,item_feat = make_test_data(False)

    # x{'user': array([0, 1, 2, 1]), 'gender': array([0, 1, 0, 1]), 'item': array([1, 2, 3, 1]), 'hist_item': array([[1, 2, 3, 0],
    #        [1, 2, 3, 0],
    #        [1, 2, 0, 0],
    #        [3, 0, 0, 0]]), 'hist_len': array([3, 3, 2, 1])}

    #y [1 1 1 1]
    # user_feat:[SparseFeat(name='user', vocabulary_size=3, embedding_dim=4, use_hash=False, vocabulary_path=None,
    # dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.RandomNormal
    # item_feat :SparseFeat(name='item', vocabulary_size=4, embedding_dim=4, use_hash=False, vocabulary_path=None,

    features = build_input_features(user_feat)
    print(features)