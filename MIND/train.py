# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/9  17:30 
# @file   ：train.PY
# @Tool   ：PyCharm

import numpy as np

from tools import *
from model import MIND
from model_list import model
def init_model(user_feature_columns,item_feature_columns):

    # 将uesr_feat 按照其类型进行划分
    item_feature_column = item_feature_columns[0]
    item_feature_name = item_feature_column.name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size
    item_embedding_dim = item_feature_columns[0].embedding_dim
    # item_index = Input(tensor=tf.constant([list(range(item_vocabulary_size))]))
    history_feature_list = [item_feature_name]
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []

    # 构建embedding时 需要所有除Densefeat以外的特征 包含user 和item
    embedding_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat) or isinstance(x,SparseFeat), user_feature_columns+item_feature_columns)) if user_feature_columns else []



    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    seq_max_len = history_feature_columns[0].maxlen

    features = build_input_features(user_feature_columns)
    item_features = build_input_features(item_feature_columns)

    inputs_list = list(features.values())
    item_inputs_list = list(item_features.values())
    inputs_list +=item_inputs_list

#[SparseFeat(name='user', vocabulary_size=3, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.RandomNormal object at 0x000001C6509AD898>, embedding_name='user', group_name='default_group', trainable=True), SparseFeat(name='gender', vocabulary_size=2, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.RandomNormal object at 0x000001C650FDE978>, embedding_name='gender', group_name='default_group', trainable=True)]
#[]
#[VarLenSparseFeat(sparsefeat=SparseFeat(name='hist_item', vocabulary_size=4, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.RandomNormal object at 0x000001C650FDEA20>, embedding_name='item', group_name='default_group', trainable=True), maxlen=4, combiner='mean', length_name='hist_len', weight_name=None, weight_norm=True)]
#[VarLenSparseFeat(sparsefeat=SparseFeat(name='hist_item', vocabulary_size=4, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.RandomNormal object at 0x000001C650FDEA20>, embedding_name='item', group_name='default_group', trainable=True), maxlen=4, combiner='mean', length_name='hist_len', weight_name=None, weight_norm=True)]
#[]
#[<tf.Tensor 'user:0' shape=(?, 1)

    # print(sparse_feature_columns)
    # print(dense_feature_columns)
    # print(varlen_sparse_feature_columns)
    print(history_feature_columns)
    # print(sparse_varlen_feature_columns)
    # print(inputs_list)
    # print(item_inputs_list)
    print(seq_max_len)
    model = MIND(user_feature_columns,item_feature_columns,features,item_features,history_feature_list,embedding_columns,
                 dense_feature_columns,sparse_varlen_feature_columns,seq_max_len)

    return model


def train(model,x,y):

    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )
    print(x,y)
    model.fit(x, y, batch_size=1, epochs=2,)
def check_model(model,x,y):
    model.fit(x, y, batch_size=10, epochs=2, validation_split=0.5)


if __name__ == '__main__':
    x, y, user_feat, item_feat = make_test_data(False)
    #model = init_model(user_feat,item_feat)
    #train(model,x,y)
    model = model(user_feat,item_feat)
    model.compile('adam', sampledsoftmaxloss)
    check_model(model, x, y)