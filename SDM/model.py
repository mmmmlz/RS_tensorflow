# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/15  15:37 
# @file   ：model.py.PY
# @Tool   ：PyCharm
from tools import *
from tensorflow.python.keras.layers import Dense
from layers import *
from tensorflow.python.keras.models import Model

def SDM(user_feature_columns,item_feature_columns,history_feature_list,num_sampled=5, units=64, rnn_layers=2,dropout_rate=0.2,
        rnn_num_res=1, num_head=4, l2_reg_embedding=1e-6, dnn_activation='tanh', seed=1024):
    item_feature_name = list(map(lambda x:x.name,item_feature_columns))
    # ['item', 'item_cate']
    item_vocabulary_size = item_feature_columns[0].vocabulary_size

    print(user_feature_columns)
    features = build_input_features(user_feature_columns)
    user_inputs_list = list(features.values())
    #print(features)
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())


    sparse_varlen_feature_columns = []
    prefer_history_columns = []
    short_history_columns = []

    prefer_fc_names = list(map(lambda x: "prefer_" + x, history_feature_list))
    short_fc_names = list(map(lambda x: "short_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in prefer_fc_names:
            prefer_history_columns.append(fc)

        elif feature_name in short_fc_names:
            short_history_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    # 这里的实现有点问题，对于embedding_name重复的feature 会覆盖之前的emd
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed)
    prefer_emb_list = embedding_lookup(embedding_matrix_dict, features, prefer_history_columns, prefer_fc_names,
                                       prefer_fc_names, to_list=True)  # L^u
    short_emb_list = embedding_lookup(embedding_matrix_dict, features, short_history_columns, short_fc_names,
                                      short_fc_names, to_list=True)  # S^u
    # dense_value_list = get_dense_input(features, dense_feature_columns)
    user_emb_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns, to_list=True)
    #print(sparse_varlen_feature_columns)

    # 对于不属于历史序列的其他变长特征单独处理，处理多值特征以及具有权重的序列特征。
    # sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, sparse_varlen_feature_columns)
    # sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
    #                                               to_list=True)

    user_emb = concat_func(user_emb_list)

    user_emb_output = Dense(units, activation=dnn_activation, name="user_emb_output")(user_emb)
    #print(user_emb_output) Tensor("user_emb_output/Tanh:0", shape=(?, 1, 64), dtype=float32)
    prefer_sess_length = features['prefer_sess_length']
    prefer_att_outputs = []
    print(prefer_emb_list)
    # 对每个长序列的feature送入Attnet
    for i, prefer_emb in enumerate(prefer_emb_list):
        prefer_attention_output = AttentionSequencePoolingLayer(dropout_rate=0)(
            [user_emb_output, prefer_emb, prefer_sess_length])
        prefer_att_outputs.append(prefer_attention_output)
    # 将长期训练特征 concat 后送入一个Dense
    prefer_att_concat = concat_func(prefer_att_outputs)
    prefer_output = Dense(units, activation=dnn_activation, name="prefer_output")(prefer_att_concat)
    #print(prefer_output)
    # shape=(?, 1, 64) prefer_output

    short_sess_length = features['short_sess_length']

    # 对于短序列，先将特征concat
    short_emb_concat = concat_func(short_emb_list)
    short_emb_input = Dense(units, activation=dnn_activation, name="short_emb_input")(short_emb_concat)
    print(short_emb_list)
    short_rnn_output = DynamicMultiRNN(num_units=units, return_sequence=True, num_layers=rnn_layers,
                                       num_residual_layers=rnn_num_res,
                                       dropout_rate=dropout_rate)([short_emb_input, short_sess_length])
    print(short_rnn_output)
    short_att_output = SelfMultiHeadAttention(num_units=units, head_num=num_head, dropout_rate=dropout_rate,
                                              future_binding=True,
                                              use_layer_norm=True)(
        [short_rnn_output, short_sess_length])  # [batch_size, time, num_units]
    # 与user embedding做 attention
   # print(short_att_output) # shape=(?, 4, 64),

    short_output = UserAttention(num_units=units, activation=dnn_activation, use_res=True, dropout_rate=dropout_rate) \
        ([user_emb_output, short_att_output, short_sess_length])
    #print(short_output)# shape=(?, 4, 64),

    gate_input = concat_func([prefer_output, short_output, user_emb_output])
    gate = Dense(units, activation='sigmoid')(gate_input)
    #print(gate)
    gate_output = Lambda(lambda x: tf.multiply(x[0], x[1]) + tf.multiply(1 - x[0], x[2]))(
        [gate, short_output, prefer_output]) 
    gate_output_reshape = Lambda(lambda x: tf.squeeze(x, 1))(gate_output)

    # 这部分是要取出所有item的embedd 用于负采样 由于item的特征有2个 需要做一个融合
    item_embedding_list = []
    item_concat = []
    for name in item_feature_name:
        item_embedding_matrix = embedding_matrix_dict[name]

        item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[name])
        item_concat.append(item_features[name])
        item_embedding_weight = NoMask()(item_embedding_matrix(item_index))
        item_embedding_list.append(item_embedding_weight)
    item_fanll = tf.reduce_mean(tf.concat(item_concat, axis=-1), axis=-1, keepdims=True)
    print("item_fanll", item_fanll) # shape=(?, 1)
    # item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])
    # item_embedding_matrix = embedding_matrix_dict[item_feature_name]
    # item_embedding_weight = NoMask()(item_embedding_matrix(item_index))

    pooling_item_embedding_weight = PoolingLayer()(item_embedding_list)
    print(pooling_item_embedding_weight)
    print(gate_output_reshape)
    output = SampledSoftmaxLayer(num_sampled=num_sampled)([
        pooling_item_embedding_weight, gate_output_reshape,item_fanll])
    print(output)
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    # model.user_input = user_inputs_list
    # model.user_embedding = gate_output_reshape

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", gate_output_reshape)

    # model.item_input = item_inputs_list
    # model.item_embedding = get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name])

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(pooling_item_embedding_weight, item_fanll))

    return model