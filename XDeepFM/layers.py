import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout,Embedding
import numpy as np
class Linear(Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        output = self.out_layer(inputs)
        return output
class Dense_layer(Layer):
    def __init__(self, hidden_units, out_dim=1, activation='relu', dropout=0.0):
        super(Dense_layer, self).__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(out_dim, activation=None)
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        # inputs: [None, n*k]
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.dropout(x)
        output = self.out_layer(x)
        return output
class CIN(Layer):
    def __init__(self, cin_size):
        super(CIN, self).__init__()
        self.cin_size = cin_size  # 每层的矩阵个数

    def build(self, input_shape):    #
        self.field_nums = [input_shape[1]]
        self.filters = []
        self.bias = []
        for i, size in enumerate(self.cin_size):
            self.filters.append(self.add_weight(
                         name='w'+str(i),
                         shape=(1, int(self.field_nums[-1] * self.field_nums[0]), size),
                         initializer=tf.initializers.glorot_uniform(),
                         regularizer=tf.keras.regularizers.l1_l2(1e-5),
                         trainable=True))

            self.bias.append(self.add_weight(
                         name='bia'+str(i),
                         shape=(size,),
                         initializer=tf.initializers.glorot_uniform(),
                         regularizer=tf.keras.regularizers.l1_l2(1e-5),
                         trainable=True))
            self.field_nums.append(size)

    def call(self, inputs, **kwargs):

        dim = int(inputs.shape[-1])
        hidden_nn_layers = [inputs]
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        # print(self.field_nums)
        for idx, layer_size in enumerate(self.cin_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
                        # [B,30,1]
            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')
            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])
            # curr_out = self.activation_layers[idx](curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
            final_result.append(curr_out)
            hidden_nn_layers.append(curr_out)
        final_result = final_result[1:]  # 去掉X0
        print(final_result)
        res = tf.concat(final_result, axis=1)  # (None, field_num[1]+...+field_num[n], k)
        output = tf.reduce_sum(res, axis=-1)  # (None, field_num[1]+...+field_num[n])

        return output
class Embedding_layer(Layer):
    def __init__(self, cate_feature_size,embedding_size):
        super(Embedding_layer, self).__init__()
        self.cate_feature_size = cate_feature_size  # 特征总数
        self.embedding_size = embedding_size

    def build(self,input_shape):
        self.emd = Embedding(self.cate_feature_size, self.embedding_size)

    def call(self, cat_index, cat_val,field_size,**kwargs):
        embeddings = self.emd(cat_index)
        feat_value = tf.reshape(cat_val, shape=[-1, field_size, 1])
            # print(feat_value.shape) (?, 30, 1)
        finall_embeddings = tf.multiply(embeddings, feat_value)

        return finall_embeddings