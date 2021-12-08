import os
from collections import OrderedDict
import pandas as pd
from config import DSIN_SESS_COUNT, DSIN_SESS_MAX_LEN, FRAC
from  model import DSIN
from tensorflow.python.keras.layers import Input
import numpy as np
import tensorflow as tf

def create_singlefeat_inputdict(feature_dim_dict, prefix=''):
    sparse_input = OrderedDict()
    for feat in feature_dim_dict["sparse"]:
        sparse_input[feat.name] = Input(
            shape=(1,), name=prefix+feat.name, dtype="float32")

    dense_input = OrderedDict()

    for  feat in feature_dim_dict["dense"]:
        dense_input[feat.name] = Input(
            shape=(1,), name=prefix+feat.name,dtype="float32")

    return sparse_input, dense_input

def get_input(feature_dim_dict, seq_feature_list, sess_max_count, seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = {}
    for idx in range(sess_max_count):
        sess_input = OrderedDict()
        for i, feat in enumerate(seq_feature_list):
            sess_input[feat] = Input(
                shape=(seq_max_len,), name='seq_' + str(idx) + str(i) + '-' + feat)
        user_behavior_input["sess_" + str(idx)] = sess_input
    user_behavior_length = {"sess_" + str(idx): Input(shape=(1,), name='seq_length' + str(idx)) for idx in
                            range(sess_max_count)}
    user_sess_length = Input(shape=(1,), name='sess_length')

    return sparse_input, dense_input, user_behavior_input, user_behavior_length, user_sess_length


if __name__ == "__main__":
    SESS_COUNT = DSIN_SESS_COUNT
    SESS_MAX_LEN = DSIN_SESS_MAX_LEN

    fd_path = './model_input/dsin_fd_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl'
    input_path = './model_input/dsin_input_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl'
    sample_sub_path = './model_input/raw_sample_'+ str(FRAC)  + '.pkl'
    label_path = './model_input/dsin_label_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl'


    fd = pd.read_pickle(fd_path)
    model_input = pd.read_pickle(input_path)
    label = pd.read_pickle(label_path)

    sample_sub = pd.read_pickle(sample_sub_path)

    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    train_idx = sample_sub.loc[sample_sub.time_stamp <1494633600, 'idx'].values[:200]
    test_idx = sample_sub.loc[sample_sub.time_stamp >=1494633600, 'idx'].values[:2100]

    train_input = [i[train_idx] for i in model_input]
    test_input = [i[test_idx] for i in model_input]

    train_label = label[train_idx]
    test_label = label[test_idx]
    #print(train_label)
    sess_count = SESS_COUNT
    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 1024

    sess_feature = ['cate_id', 'brand']
    spare_list = sess_feature
    TEST_BATCH_SIZE = 2 ** 14

    train_fin = []
    for t in train_input:
        q = t.astype(float)
        train_fin.append(q)
    #print(train_fin[0])
    sparse_input, dense_input, user_behavior_input_dict, _, user_sess_length = get_input(
        fd, sess_feature, sess_count, sess_len_max)
    #user_sess_length:针对每一个用户，他的sess个数 最小为0 最大为5 因为一个用户如果有行为满足一个sess的要求，会进行padding满一个sess。

    # 将输入都转换成了tensor

    #print(sparse_input)

    model = DSIN(fd,sess_feature, embedding_size=4, sess_max_count=sess_count,
                 sess_len_max=sess_len_max, dnn_hidden_units=(200, 80), att_head_num=8,att_embedding_size=1,spare_list=spare_list)
    model.compile("adam", "binary_crossentropy",
                    metrics=['binary_crossentropy'], )
    x_train = {"sparse_input":sparse_input,
               "dense_input":dense_input,
               "user_behavior_input_dict":user_behavior_input_dict,
               "user_sess_length":user_sess_length}
    y_train_ = tf.convert_to_tensor(np.array(label))

    model.fit(train_fin,train_label)