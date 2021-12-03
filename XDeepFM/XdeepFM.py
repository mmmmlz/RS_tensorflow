import tensorflow as tf
from layers import Dense_layer,CIN,Linear,Embedding_layer
from sklearn.metrics import roc_auc_score
import config
from tensorflow.keras.layers import Dense, Embedding

import numpy as np





class XdeepFm(tf.keras.Model):

    def __init__(self,cate_feature_size, field_size, numeric_feature_size,
                 embedding_size=8,
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 greater_is_better=True,cin_layer=[124,124]):
        super(XdeepFm, self).__init__()
        self.cate_feature_size = cate_feature_size
        self.numeric_feature_size = numeric_feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.total_size = self.field_size * self.embedding_size + self.numeric_feature_size
        self.deep_layers = deep_layers
        self.cin_layer = cin_layer
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []
        self.cross_layer_num = len(self.cin_layer)

        self.embeding_weights = dict()

        # embeddings
        # self.embeding_weights['feature_embeddings'] = tf.Variable(
        #     tf.random_normal([self.cate_feature_size, self.embedding_size], 0.0, 0.01,dtype=tf.float32),
        #     name='feature_embeddings2',trainable=True,dtype=tf.float32)
       # self.emd = Embedding(self.cate_feature_size,self.embedding_size)
        # self.embeding_weights['feature_bias'] = tf.Variable(tf.random_normal([self.cate_feature_size, 1], 0.0, 1.0),
        #                                       name='feature_bias',trainable=True)
        self.emd = Embedding_layer(self.cate_feature_size,self.embedding_size)

        # DNN layer
        self.Dnn_layer = Dense_layer(deep_layers,1,activation='relu', dropout=0.0)

        # Liner layer
        self.Liner_layer = Linear()

        # CIN layer
        self.Cin_layer = CIN(self.cin_layer)
        # out layer
        self.out_layer = Dense(1, activation=None)


    def call(self, inputs):
        cat_index,cat_val,numeric = inputs["cate_idx"],inputs["cate_value"],inputs["numeric"]
        print(cat_index.shape,cat_val.shape,numeric.shape)

        #embeddings = tf.nn.embedding_lookup(self.embeding_weights['feature_embeddings'], cat_index)

        # embedding layer
        # embeddings = self.emd(cat_index)
        # print("emb",embeddings.shape)
        #     # embeddings [B,30,8]
        # feat_value = tf.reshape(cat_val, shape=[-1, self.field_size, 1])
        #     # print(feat_value.shape) (?, 30, 1)
        finall_embeddings = self.emd(cat_index,cat_val,self.field_size)

        x0 = tf.concat([numeric,
                        tf.reshape(finall_embeddings, shape=[-1, self.field_size * self.embedding_size])]
                       , axis=1)
            # print(x0.shape) (?, 249)
        liner_out = self.Liner_layer(x0)
        #print(liner_out.shape)  [B,1]
        dnn_out = self.Dnn_layer(x0)
        # print(dnn_out) shape=(?, 1)
        cin_out = self.Cin_layer(finall_embeddings)
        #print(cin_out.shape) (?, 124)
        output = self.out_layer(liner_out + cin_out + dnn_out)
        print(output.shape)

        return output


if __name__ == '__main__':
    XdeepFM_params = {
        "embedding_size": 8,
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "random_seed": config.RANDOM_SEED,
        "cin_layer": [124, 124]
    }
    XdeepFM_params["cate_feature_size"] = 3
    XdeepFM_params["field_size"] = 4
    XdeepFM_params['numeric_feature_size'] = 5
    print(XdeepFM_params)
    a = XdeepFm(**XdeepFM_params)
    x_train = [[11,12,13],[21,22,23],[31,32,33]]
    y_train = [[1],[0],[1]]
    x2_train = []