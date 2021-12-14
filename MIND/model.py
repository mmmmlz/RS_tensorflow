# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/9  16:17 
# @file   ：model.py
# @Tool   ：PyCharm


from tensorflow.python.keras.models import Model

from layer import *
from tools import ALL_FEATURES,ITEM_FEATURES,SPARSE_FEATURES,DENSE_FEATURES
from tools import pooling


class MIND(Model):
    def __init__(self,user_feature_columns,item_feature_columns,features,item_features,history_feature_list,embedding_columns,
                 dense_feature_columns,sparse_varlen_feature_columns,seq_max_len, num_sampled=5, k_max=2, p=1.0, dynamic_k=False,
           user_dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6,
           dnn_dropout=0, output_activation='linear', seed=1024):
        super(MIND,self).__init__()

        self.item_feature_columns = item_feature_columns
        self.item_embedding_dim = item_feature_columns[0].embedding_dim

        self.embedding_layer = Embedding_layer(embedding_columns,l2_reg_dnn,seed=seed,stag="sparse_embedding")

        self.capsule_layer = CapsuleLayer(self.item_embedding_dim,seq_max_len)

        self.dnn_layer = Dense_layer(user_dnn_hidden_units)


    def call(self, input):
        item_features, user_features, dense_features = {}, {}, {}
        for feat in ALL_FEATURES:
            if feat in ITEM_FEATURES:
                item_features[feat] = input[feat]
            elif feat in DENSE_FEATURES:
                dense_features[feat] = input[feat]
            else:
                user_features[feat] = input[feat]
        # get embedding
        hist_len = user_features['hist_len']

        flag = 1
        query_emb_list = self.embedding_layer(item_features, feat_columns=ITEM_FEATURES, flag=flag)
        # [[B,4]]
        print(query_emb_list)
        flag = 2
        # 对于历史行为 需要的获得的embedding其实也是历史行为中的item 只不过是从user_features里面获得
        keys_emb_list = self.embedding_layer(user_features, feat_columns=ITEM_FEATURES, flag=flag)
        # [[B,4,4]]
        print(keys_emb_list)
        # 之前是将每个用户每个行为的不同特征（item_id,cate_id_brand_id等）进行concat，在MIND中是进行POOLING
        # Polling的方法，对于列表中每一个[4,4] 先在最后一维膨胀[4,4,1] 再concat[4,4,len(list)] 最后在最后一维做reduce_mean

        # print(keys_emb_list)
        keys_emb = pooling(keys_emb_list)
        query_emb = pooling(query_emb_list,flag="eeee2")
        flag = 1
        # 这里注意不要送入his_len 其没有embedding 仅用于mask
        # dnn_input_emb_list = self.embedding_layer(user_features, feat_columns=SPARSE_FEATURES, flag=flag)
        # dnn_input_emb = tf.concat(dnn_input_emb_list,axis=-1,name="concat_3")
        # [B,8]
        print(keys_emb)  # [B,4,4]
        #print(squash(keys_emb))
        print(hist_len)
        # 构建胶囊网络
        out_cap = self.capsule_layer((keys_emb, hist_len))
        print("out_cat",out_cap)# (?, 2, 2)

        for k,v in dense_features.items():
            t = tf.reshape(v,[-1,1])
            t = tf.expand_dims(t,-1)
            t = tf.tile(t,[1,tf.shape(out_cap)[1],tf.shape(out_cap)[2]])
            dnn_in = tf.concat([t,out_cap],axis=-1)
        dnn_in = tf.reshape(dnn_in,[-1,4,8])
        user_embedding = self.dnn_layer(dnn_in)

        #user_embedding = tf.reshape(user_embedding,[-1,4])
        user_embedding = user_embedding + tf.expand_dims(query_emb,-1)
        user_embedding = tf.reduce_sum(user_embedding,axis=1,name="finall")

        print(user_embedding)
        return user_embedding









        #print(query_emb_list)

        return










        return


