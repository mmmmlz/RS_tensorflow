# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/28  9:31 
# @file   ：model.PY
# @Tool   ：PyCharm


from tools import *
from layer import *

from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Flatten, Input)


def BST(dnn_feature_columns, history_feature_list, transformer_num=1, att_head_num=4,
        use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu', l2_reg_dnn=0,
        l2_reg_embedding=1e-6, dnn_dropout=0.0, seed=1024, task='binary'):
    #print(dnn_feature_columns)

    #print(history_feature_list)

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    user_behavior_length = features["seq_length"]

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="",
                                             seq_mask_zero=True)
    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                      return_feat_list=history_feature_list, to_list=True)
    hist_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns,
                                     return_feat_list=history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)

    dense_value_list = get_dense_input(features, dense_feature_columns)

    query_emb = concat_func(query_emb_list)
    deep_input_emb = concat_func(dnn_input_emb_list)
    hist_emb = concat_func(hist_emb_list)

    print(dnn_input_emb_list)
    print(query_emb)
    print(deep_input_emb)
    print(hist_emb)
    transformer_output = hist_emb
    for _ in range(transformer_num):
        att_embedding_size = transformer_output.get_shape().as_list()[-1] // att_head_num
        print(att_embedding_size)
        transformer_layer = Transformer(att_embedding_size=att_embedding_size, head_num=att_head_num,
                                        dropout_rate=dnn_dropout, use_positional_encoding=True, use_res=True,
                                        use_feed_forward=True, use_layer_norm=True, blinding=False, seed=seed,
                                        supports_masking=False, output_type=None)
        transformer_output = transformer_layer([transformer_output, transformer_output,
                                                user_behavior_length, user_behavior_length])
        print(transformer_output)
    attn_output = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                supports_masking=False)([query_emb, transformer_output,
                                                                         user_behavior_length])
    print(attn_output) #(?, 1, 12),
    deep_input_emb = concat_func([deep_input_emb, attn_output], axis=-1)
    #Tensor("concatenate_3/concat:0", shape=(?, 1, 38), dtype=float32)
    #    Tensor("flatten/Reshape:0", shape=(?, 38), dtype=float32)
    deep_input_emb = Flatten()(deep_input_emb)
    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(dnn_input)
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)


    return model