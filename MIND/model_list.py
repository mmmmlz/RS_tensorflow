# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/13  20:43 
# @file   ：model_list.PY
# @Tool   ：PyCharm
from tensorflow.python.keras.models import Model

from layer import *
from tools import *
from tools import pooling
from feature_column import SparseFeat,VarLenSparseFeat,DenseFeat
from tensorflow.python.keras.layers import Concatenate
# jiang dnn 部分的输入第二维度膨胀到和胶囊网络出来的胶囊个数一样
def tile_user_otherfeat(user_other_feature, k_max):
    return tf.tile(tf.expand_dims(user_other_feature, -2), [1, k_max, 1])

def model(user_feature_columns, item_feature_columns, num_sampled=2, k_max=6, p=1.0, dynamic_k=False,
         user_dnn_hidden_units=(32, 4), dnn_activation='relu', dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6,
         dnn_dropout=0, output_activation='linear', seed=1024):
    print(user_feature_columns)
    # if len(item_feature_columns) > 1:
    #     raise ValueError("Now MIND only support 1 item feature like item_id")
    item_feature_column = item_feature_columns[0]
    item_feature_name = list(map(lambda x:x.name, item_feature_columns))
    item_vocabulary_size = item_feature_columns[0].vocabulary_size
    item_embedding_dim = item_feature_columns[0].embedding_dim
    # item_index = Input(tensor=tf.constant([list(range(item_vocabulary_size))]))

    history_feature_list = item_feature_name

    features = build_input_features(user_feature_columns)
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []
    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        #print(feature_name)
        if feature_name in history_fc_names:
            #print(feature_name)
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    seq_max_len = history_feature_columns[0].maxlen
    inputs_list = list(features.values())


    print("inputs_list",inputs_list)
    print("user_feature_columns",user_feature_columns)
    print("item_feature_columns",item_feature_columns)
    print("history_feature_list",history_feature_list)
    print("history_feature_columns",history_feature_columns)
    print("history_fc_names",history_fc_names)
    print("varlen_sparse_feature_columns",varlen_sparse_feature_columns)
    print("dense_feature_columns",dense_feature_columns)  #[DenseFeat(name='price', dimension=1, dtype='float32', transform_fn=None)]
    print("sparse_varlen_feature_columns",sparse_varlen_feature_columns)


    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns,
                                                    l2_reg_embedding,seed=seed, prefix="")
    '''
        {'gender': <tensorflow.python.keras.layers.embeddings.Embedding at 0x1e7e019e6a0>,
         'i_brand_id': <tensorflow.python.keras.layers.embeddings.Embedding at 0x1e7e019e1d0>,
         'i_cate_id': <tensorflow.python.keras.layers.embeddings.Embedding at 0x1e7e019e710>,
         'item': <tensorflow.python.keras.layers.embeddings.Embedding at 0x1e7e019e3c8>,
         'user': <tensorflow.python.keras.layers.embeddings.Embedding at 0x1e7e0364898>}
    '''
    item_features = build_input_features(item_feature_columns)
    # item_features OrderedDict([('item', <tf.Tensor 'item:0' shape=(?, 1) dtype=int32>),
    # ('i_cate_id', <tf.Tensor 'i_cate_id:0' shape=(?, 1) dtype=int32>),
    # ('i_brand_id', <tf.Tensor 'i_brand_id:0' shape=(?, 1) dtype=int32>)])

    query_emb_list = embedding_lookup(embedding_matrix_dict, item_features, item_feature_columns,
                                      history_feature_list,
                                       history_feature_list, to_list=True)
    # [<tf.Tensor 'sparse_seq_emb_hist_brand/embedding_lookup/Identity_1:0' shape=(?, 1, 4) dtype=float32>,
    # <tf.Tensor 'sparse_emb_i_cate_id/embedding_lookup/Identity_1:0' shape=(?, 1, 4) dtype=float32>,
    # <tf.Tensor 'sparse_emb_i_brand_id/embedding_lookup/Identity_1:0' shape=(?, 1, 4) dtype=float32>]

    keys_emb_list = embedding_lookup(embedding_matrix_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)


    # [<tf.Tensor 'sparse_seq_emb_hist_i_brand_id_1/embedding_lookup/Identity_1:0' shape=(?, 4, 4) dtype=float32>,
    # <tf.Tensor 'sparse_seq_emb_hist_i_brand_id_2/embedding_lookup/Identity_1:0' shape=(?, 4, 4) dtype=float32>,
    # <tf.Tensor 'sparse_seq_emb_hist_i_brand_id_3/embedding_lookup/Identity_1:0' shape=(?, 4, 4) dtype=float32>]

    dnn_input_emb_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    #[<tf.Tensor 'sparse_emb_user/embedding_lookup/Identity_1:0' shape=(?, 1, 4) dtype=float32>,
    # <tf.Tensor 'sparse_emb_gender/embedding_lookup/Identity_1:0' shape=(?, 1, 4) dtype=float32>]

    dense_value_list = get_dense_input(features, dense_feature_columns)
     # [<tf.Tensor 'price:0' shape=(?, 1) dtype=float32>]
    history_emb = PoolingLayer()(NoMask()(keys_emb_list))  # 这里是因为PoolingLayer 的supports_masking=False 需要通过Nomask处理
    target_emb = PoolingLayer()(NoMask()(query_emb_list))
     #Tensor("pooling_layer/Mean:0", shape=(?, 4, 4), dtype=float32)
    # Tensor("pooling_layer_1/Mean:0", shape=(?, 1, 4), dtype=float32)
    hist_len = features['hist_len']
    # Tensor("hist_len_2:0", shape=(?, 1), dtype=int32)

    high_capsule = CapsuleLayer(input_units=item_embedding_dim,
                                out_units=item_embedding_dim, max_len=seq_max_len,
                                k_max=k_max)((history_emb, hist_len))
    print(high_capsule)
    # Tensor("capsule_layer/Reshape:0", shape=(?, 6, 4), dtype=float32)
    if len(dnn_input_emb_list) > 0 or len(dense_value_list) > 0:
        user_other_feature = combined_dnn_input(dnn_input_emb_list, dense_value_list) # Tensor("concatenate_3/concat:0", shape=(?, 9), dtype=float32)
        other_feature_tile = tf.keras.layers.Lambda(tile_user_otherfeat, arguments={'k_max': k_max})(user_other_feature) # Tensor("lambda/Tile:0", shape=(?, 6, 9), dtype=float32)

        #user_deep_input = Concatenate()([NoMask()(other_feature_tile), high_capsule])
        user_deep_input = tf.concat([NoMask()(other_feature_tile), high_capsule],axis=-1)

        print(user_deep_input)#Tensor("concatenate_4/concat:0", shape=(?, 6, 13), dtype=float32)
    else:
        user_deep_input = high_capsule

    user_embeddings = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn,
                          dnn_dropout, dnn_use_bn, output_activation=output_activation, seed=seed,
                          name="user_embedding")(user_deep_input)
    # Tensor("user_embedding/dropout_1/cond/Merge:0", shape=(?, 6, 4), dtype=float32)

    item_inputs_list = list(item_features.values())
    # [<tf.Tensor 'item:0' shape=(?, 1) dtype=int32>, <tf.Tensor 'i_cate_id:0' shape=(?, 1) dtype=int32>, <tf.Tensor 'i_brand_id:0' shape=(?, 1) dtype=int32>]

    # 这部分是要取出所有item的embedd 用于负采样 由于item的特征有三个 需要做一个融合
    item_embedding_list = []
    item_concat = []
    for name in item_feature_name:
        item_embedding_matrix = embedding_matrix_dict[name]

        item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[name])
        item_concat.append(item_features[name])
        item_embedding_weight = NoMask()(item_embedding_matrix(item_index))
        item_embedding_list.append(item_embedding_weight)

    item_fanll = tf.reduce_mean(tf.concat(item_concat,axis=-1),axis=-1,keepdims=True)
    print("item_fanll",item_fanll)
    pooling_item_embedding_weight = PoolingLayer()(item_embedding_list)
    print(pooling_item_embedding_weight) #shape=(4, 4)
    if dynamic_k:
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p, )((user_embeddings, target_emb, hist_len))
    else:
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p, )((user_embeddings, target_emb))
    # k个胶囊进行了加权和 shape=(?, 4) user_embedding_final

    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [pooling_item_embedding_weight, user_embedding_final, item_fanll])
    print("item_feature_name[0]",item_features[item_feature_name[0]])
    model = Model(inputs=inputs_list + item_inputs_list, outputs=output)
    model.__setattr__("user_input", inputs_list)
    model.__setattr__("user_embedding", user_embeddings)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",get_item_embedding(pooling_item_embedding_weight, item_fanll))

    return model