# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/15  15:38 
# @file   ：layers.py.PY
# @Tool   ：PyCharm

import tensorflow as tf
from tensorflow.python.ops.lookup_ops import TextFileInitializer
from tensorflow.python.keras.initializers import TruncatedNormal,Ones, Zeros
from tensorflow.python.keras import backend as K
from tools import *

try:
    from tensorflow.python.ops.lookup_ops import StaticHashTable
except ImportError:
    from tensorflow.python.ops.lookup_ops import HashTable as StaticHashTable

from tensorflow.python.keras.layers import Dense,Dropout


class Hash(tf.keras.layers.Layer):
    """Looks up keys in a table when setup `vocabulary_path`, which outputs the corresponding values.
    If `vocabulary_path` is not set, `Hash` will hash the input to [0,num_buckets). When `mask_zero` = True,
    input value `0` or `0.0` will be set to `0`, and other value will be set in range [1,num_buckets).

    The following snippet initializes a `Hash` with `vocabulary_path` file with the first column as keys and
    second column as values:

    * `1,emerson`
    * `2,lake`
    * `3,palmer`

    # >>> hash = Hash(
    # ...   num_buckets=3+1,
    # ...   vocabulary_path=filename,
    # ...   default_value=0)
    # >>> hash(tf.constant('lake')).numpy()
    # 2
    # >>> hash(tf.constant('lakeemerson')).numpy()
    # 0

    Args:
        num_buckets: An `int` that is >= 1. The number of buckets or the vocabulary size + 1
            when `vocabulary_path` is setup.
        mask_zero: default is False. The `Hash` value will hash input `0` or `0.0` to value `0` when
            the `mask_zero` is `True`. `mask_zero` is not used when `vocabulary_path` is setup.
        vocabulary_path: default `None`. The `CSV` text file path of the vocabulary hash, which contains
            two columns seperated by delimiter `comma`, the first column is the value and the second is
            the key. The key data type is `string`, the value data type is `int`. The path must
            be accessible from wherever `Hash` is initialized.
        default_value: default '0'. The default value if a key is missing in the table.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, num_buckets, mask_zero=False, vocabulary_path=None, default_value=0, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        self.vocabulary_path = vocabulary_path
        self.default_value = default_value
        if self.vocabulary_path:
            initializer = TextFileInitializer(vocabulary_path, 'string', 1, 'int64', 0, delimiter=',')
            self.hash_table = StaticHashTable(initializer, default_value=self.default_value)
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):

        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        if self.vocabulary_path:
            hash_x = self.hash_table.lookup(x)
            return hash_x

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                                   name=None)  # weak hash
        except AttributeError:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, 'vocabulary_path': self.vocabulary_path,
                  'default_value': self.default_value}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None

class ConcatAttention(tf.keras.layers.Layer):
    """
    :param query: [batch_size, T, C_q]
    :param key:   [batch_size, T, C_k]
    :return:      [batch_size, 1, T]
        query_size should keep the same dim with key_size
    """

    def __init__(self, scale=True, **kwargs):
        self.scale = scale
        super(ConcatAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ConcatAttention` layer should be called '
                             'on a list of 2 tensors')
        self.projection_layer = Dense(units=1, activation='tanh')
        super(ConcatAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        query, key = inputs
        q_k = tf.concat([query, key], axis=-1)
        output = self.projection_layer(q_k)
        if self.scale == True:
            output = output / (key.get_shape().as_list()[-1] ** 0.5)
        output = tf.transpose(output, [0, 2, 1])
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def compute_mask(self, inputs, mask):
        return mask
class AttentionSequencePoolingLayer(tf.keras.layers.Layer):
    """
    :param query:           [batch_size, 1, C_q]
    :param keys:            [batch_size, T, C_k]
    :param keys_length:      [batch_size, 1]
    :return:                [batch_size, 1, C_k]
    """

    def __init__(self, dropout_rate=0, **kwargs):
        self.dropout_rate = dropout_rate
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `SequenceFeatureMask` layer should be called '
                             'on a list of 3 inputs')
        self.concat_att = ConcatAttention()
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, future_binding=False)
        super(AttentionSequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        queries, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)
        queries = tf.tile(queries, [1, hist_len, 1])  # [batch_size, T, units]
        attention_score = self.concat_att([queries, keys])  # [batch_size, 1, units]

        outputs = self.softmax_weight_sum([attention_score, keys, key_masks])
        # [batch_size, units]
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask


class SoftmaxWeightedSum(tf.keras.layers.Layer):
    """
    :param align:           [batch_size, 1, T]
    :param value:           [batch_size, T, units]
    :param key_masks:       [batch_size, 1, T]
                            2nd dim size with align
    :param drop_out:
    :param future_binding:
    :return:                weighted sum vector
                            [batch_size, 1, units]
    """

    def __init__(self, dropout_rate=0.2, future_binding=False, seed=2020, **kwargs):
        self.dropout_rate = dropout_rate
        self.future_binding = future_binding
        self.seed = seed
        super(SoftmaxWeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `SoftmaxWeightedSum` layer should be called '
                             'on a list of 3 tensors')
        if input_shape[0][-1] != input_shape[2][-1]:
            raise ValueError('query_size should keep the same dim with key_mask_size')
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        super(SoftmaxWeightedSum, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        align, value, key_masks = inputs
        paddings = tf.ones_like(align) * (-2 ** 32 + 1)
        align = tf.where(key_masks, align, paddings)
        if self.future_binding:
            length = value.get_shape().as_list()[1]
            lower_tri = tf.ones([length, length])
            try:
                lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
            except:
                lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
            masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
            align = tf.where(tf.equal(masks, 0), paddings, align)

        align = tf.nn.softmax(align)

        align = self.dropout(align, training=training)
        output = tf.matmul(align, value)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate, 'future_binding': self.future_binding}
        base_config = super(SoftmaxWeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask

class DynamicMultiRNN(tf.keras.layers.Layer):
    def __init__(self, num_units=None, rnn_type='LSTM', return_sequence=True, num_layers=2, num_residual_layers=1,
                 dropout_rate=0.2,
                 forget_bias=1.0, **kwargs):

        self.num_units = num_units
        self.return_sequence = return_sequence
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_residual_layers = num_residual_layers
        self.dropout = dropout_rate
        self.forget_bias = forget_bias
        super(DynamicMultiRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]

        # 构建LSTM三部曲 1 创建Cell
        if self.rnn_type == "LSTM":
            try:
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
            except AttributeError:
                single_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)

        elif self.rnn_type == "GRU":
            try:
                single_cell = tf.nn.rnn_cell.GRUCell(self.num_units, forget_bias=self.forget_bias)
            except AttributeError:
                single_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.num_units, forget_bias=self.forget_bias)
        else:
            raise ValueError("Unknown unit type %s!" % self.rnn_type)

        # 2.1 设置dropout
        dropout = self.dropout if tf.keras.backend.learning_phase() == 1 else 0
        try:
            single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        except AttributeError:
            single_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        cell_list = []
        # 2.2 设置 残差
        for i in range(self.num_layers):
            residual = (i >= self.num_layers - self.num_residual_layers)
            if residual:
                try:
                    single_cell_residual = tf.nn.rnn_cell.ResidualWrapper(single_cell)
                except AttributeError:
                    single_cell_residual = tf.compat.v1.nn.rnn_cell.ResidualWrapper(single_cell)
                cell_list.append(single_cell_residual)
            else:
                cell_list.append(single_cell)
        # 3 连接多层

        if len(cell_list) == 1:
            self.final_cell = cell_list[0]
        else:
            try:
                self.final_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
            except AttributeError:
                self.final_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_list)
        super(DynamicMultiRNN, self).build(input_shape)

    def call(self, input_list, mask=None, training=None):
        rnn_input, sequence_length = input_list

        try:
            with tf.name_scope("rnn"), tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
                rnn_output, hidden_state = tf.nn.dynamic_rnn(self.final_cell, inputs=rnn_input,
                                                             sequence_length=tf.squeeze(sequence_length),
                                                             dtype=tf.float32, scope=self.name)
        except AttributeError:
            with tf.name_scope("rnn"), tf.compat.v1.variable_scope("rnn", reuse=tf.compat.v1.AUTO_REUSE):
                rnn_output, hidden_state = tf.compat.v1.nn.dynamic_rnn(self.final_cell, inputs=rnn_input,
                                                                       sequence_length=tf.squeeze(sequence_length),
                                                                       dtype=tf.float32, scope=self.name)
        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(hidden_state, axis=1)

    def compute_output_shape(self, input_shape):
        rnn_input_shape = input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return (None, 1, rnn_input_shape[2])

    def get_config(self, ):
        config = {'num_units': self.num_units, 'rnn_type': self.rnn_type, 'return_sequence': self.return_sequence,
                  'num_layers': self.num_layers,
                  'num_residual_layers': self.num_residual_layers, 'dropout_rate': self.dropout, 'forget_bias':self.forget_bias}
        base_config = super(DynamicMultiRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SelfMultiHeadAttention(tf.keras.layers.Layer):
    """
      :param query: A 3d tensor with shape of [batch_size, T, C]
      :param key_masks: A 3d tensor with shape of [batch_size, 1]
      :return: A 3d tensor with shape of  [batch_size, T, C]
    """

    def __init__(self, num_units=8, head_num=4, scale=True, dropout_rate=0.2, future_binding=True, use_layer_norm=True,
                 use_res=True,
                 seed=2020, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.num_units = num_units
        self.head_num = head_num
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.future_binding = future_binding
        self.use_layer_norm = use_layer_norm
        self.use_res = use_res
        self.seed = seed
        super(SelfMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `SelfMultiHeadAttention` layer should be called '
                             'on a list of 2 tensors')
        if len(input_shape[0]) != 3 or len(input_shape[1]) != 2:
            raise ValueError('input: [N, T_k, d_model], key masks: [N, key_seqlen]')
        embedding_size = int(input_shape[0][-1])
        if self.num_units == None:
            self.num_units = embedding_size
        self.W = self.add_weight(name='Q_K_V', shape=[embedding_size, self.num_units * 3],
                                 dtype=tf.float32,
                                 initializer=TruncatedNormal(seed=self.seed))
        self.W_output = self.add_weight(name='output_W', shape=[self.num_units, self.num_units],
                                        dtype=tf.float32,
                                        initializer=TruncatedNormal(seed=self.seed))

        self.layer_norm = LayerNormalization()
        self.attention = DotAttention(scale=self.scale)
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, future_binding=self.future_binding,
                                                     seed=self.seed)
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.seq_len_max = int(input_shape[0][1])
        # Be sure to call this somewhere!
        super(SelfMultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        input_info, keys_length = inputs

        hist_len = input_info.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)
        key_masks = tf.squeeze(key_masks, axis=1)

        Q_K_V = tf.tensordot(input_info, self.W, axes=(-1, 0))  # [N T_q D*3]
        querys, keys, values = tf.split(Q_K_V, 3, -1)

        # head_num None F D
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)  # (h*N, T_q, C/h)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)  # (h*N, T_k, C/h)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)  # (h*N, T_k, C/h)
        print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
        # print(querys)
        # print(input_info)
        # print(key_masks)
        # (h*N, T_q, T_k)
        align = self.attention([querys, keys])
        #print(align) # shape=(?, 4, 4)
        key_masks = tf.tile(key_masks, [self.head_num, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(input_info)[1], 1])  # (h*N, T_q, T_k)

        outputs = self.softmax_weight_sum([align, values, key_masks])  # (h*N, T_q, C/h)
        outputs = tf.concat(tf.split(outputs, self.head_num, axis=0), axis=2)  # (N, T_q, C)

        outputs = tf.tensordot(outputs, self.W_output, axes=(-1, 0))  # (N, T_q, C)
        outputs = self.dropout(outputs, training=training)
        if self.use_res:
            outputs += input_info
        if self.use_layer_norm:
            outputs = self.layer_norm(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.num_units)

    def get_config(self, ):
        config = {'num_units': self.num_units, 'head_num': self.head_num, 'scale': self.scale,
                  'dropout_rate': self.dropout_rate,
                  'future_binding': self.future_binding, 'use_layer_norm': self.use_layer_norm, 'use_res': self.use_res,
                  'seed': self.seed}
        base_config = super(SelfMultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask

class LayerNormalization(tf.keras.layers.Layer):

    # LN 在最后一维度做LN
    def __init__(self, axis=-1, eps=1e-9, center=True,
                 scale=True, **kwargs):
        self.axis = axis
        self.eps = eps
        self.center = center
        self.scale = scale
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.eps)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'eps': self.eps, 'center': self.center, 'scale': self.scale}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DotAttention(tf.keras.layers.Layer):
    """
    :param query: [batch_size, 1, C]
    :param key:   [batch_size, T, C]
    :return:      [batch_size, 1, T]
    """

    def __init__(self, scale=True, **kwargs):
        self.scale = scale
        super(DotAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `DotAttention` layer should be called '
                             'on a list of 2 tensors')
        if input_shape[0][-1] != input_shape[1][-1]:
            raise ValueError('query_size should keep the same dim with key_size')
        super(DotAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        query, key = inputs
        output = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        if self.scale == True:
            output = output / (key.get_shape().as_list()[-1] ** 0.5)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

    def compute_mask(self, inputs, mask):
        return mask

class UserAttention(tf.keras.layers.Layer):
    """
      :param query: A 3d tensor with shape of [batch_size, T, C]
      :param keys: A 3d tensor with shape of [batch_size, T, C]
      :param key_masks: A 3d tensor with shape of [batch_size, 1]
      :return: A 3d tensor with shape of  [batch_size, 1, C]
    """

    def __init__(self, num_units=None, activation='tanh', use_res=True, dropout_rate=0, scale=True, seed=2020,
                 **kwargs):
        self.scale = scale
        self.num_units = num_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.use_res = use_res
        super(UserAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `UserAttention` layer should be called '
                             'on a list of 3 tensors')
        if self.num_units == None:
            self.num_units = input_shape[0][-1]
        self.dense = Dense(self.num_units, activation=self.activation)
        self.attention = DotAttention(scale=self.scale)
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, seed=self.seed)
        super(UserAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        user_query, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)
        query = self.dense(user_query)

        align = self.attention([query, keys])

        output = self.softmax_weight_sum([align, keys, key_masks])

        if self.use_res:
            output += keys
        return tf.reduce_mean(output, 1, keep_dims=True)

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][2])

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):
        config = {'num_units': self.num_units, 'activation': self.activation, 'use_res': self.use_res,
                  'dropout_rate': self.dropout_rate,
                  'scale': self.scale, 'seed': self.seed, }
        base_config = super(UserAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbeddingIndex(tf.keras.layers.Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SampledSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        embeddings, inputs, label_idx = inputs_with_label_idx

        loss = tf.nn.sampled_softmax_loss(weights=embeddings,  # self.item_embedding.
                                          biases=self.zero_bias,
                                          labels=label_idx,
                                          inputs=inputs,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.size,  # self.target_song_size
                                          )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'num_sampled': self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PoolingLayer(tf.keras.layers.Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):

        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = tf.keras.layers.Concatenate(axis=-1)(expand_seq_value_len_list)
        if self.mode == "mean":
            hist = tf.reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = tf.reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = tf.reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))