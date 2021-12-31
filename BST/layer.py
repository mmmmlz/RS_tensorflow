# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/28  9:32 
# @file   ：layer.PY
# @Tool   ：PyCharm
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import numpy as np
from tensorflow.python.keras.regularizers import l2

from tensorflow.python.keras.initializers import Ones,Zeros
from tensorflow.python.ops.init_ops import GlorotNormal
from tensorflow.python.ops.init_ops import GlorotUniform
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform

def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)
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



class Transformer(Layer):
    """  Simplified version of Transformer  proposed in 《Attention is all you need》

      Input shape
        - a list of two 3D tensor with shape ``(batch_size, timesteps, input_dim)`` if ``supports_masking=True`` .
        - a list of two 4 tensors, first two tensors with shape ``(batch_size, timesteps, input_dim)``,last two tensors with shape ``(batch_size, 1)`` if ``supports_masking=False`` .


      Output shape
        - 3D tensor with shape: ``(batch_size, 1, input_dim)``  if ``output_type='mean'`` or ``output_type='sum'`` , else  ``(batch_size, timesteps, input_dim)`` .


      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **dropout_rate**: float between 0 and 1. Fraction of the units to drop.
            - **use_positional_encoding**: bool. Whether or not use positional_encoding
            - **use_res**: bool. Whether or not use standard residual connections before output.
            - **use_feed_forward**: bool. Whether or not use pointwise feed foward network.
            - **use_layer_norm**: bool. Whether or not use Layer Normalization.
            - **blinding**: bool. Whether or not use blinding.
            - **seed**: A Python integer to use as random seed.
            - **supports_masking**:bool. Whether or not support masking.
            - **attention_type**: str, Type of attention, the value must be one of { ``'scaled_dot_product'`` , ``'additive'`` }.
            - **output_type**: ``'mean'`` , ``'sum'`` or `None`. Whether or not use average/sum pooling for output.

      References
            - [Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
    """

    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=False, blinding=True, seed=1024, supports_masking=False,
                 attention_type="scaled_dot_product", output_type="mean", **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        super(Transformer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                    self.att_embedding_size, self.head_num, embedding_size))
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
        if self.attention_type == "additive":
            self.b = self.add_weight('b', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.v = self.add_weight('v', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        # if self.use_res:
        #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
        #                                  initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))

        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()
        if self.use_positional_encoding:
            self.query_pe = PositionEncoding()
            self.key_pe = PositionEncoding()
        # Be sure to call this somewhere!
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs

            query_masks = tf.sequence_mask(
                query_masks, self.seq_len_max, dtype=tf.float32)
            key_masks = tf.sequence_mask(
                key_masks, self.seq_len_max, dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)

        if self.use_positional_encoding:
            queries = self.query_pe(queries)
            keys = self.key_pe(queries)
        # 使用 tensordot 进行不同维度的矩阵相乘 [B,T,D]x[D,num*D] = [B,T,num*D]
        querys = tf.tensordot(queries, self.W_Query,
                              axes=1)  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))
        print("q",querys,"k",keys,"v",values)
        # head_num*None T_q D
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":
            # head_num*None T_q T_k.
            outputs = tf.matmul(querys, keys, transpose_b=True)

            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        elif self.attention_type == "additive":
            querys_reshaped = tf.expand_dims(querys, axis=-2)
            keys_reshaped = tf.expand_dims(keys, axis=-3)
            outputs = tf.tanh(tf.nn.bias_add(querys_reshaped + keys_reshaped, self.b))
            outputs = tf.squeeze(tf.tensordot(outputs, tf.expand_dims(self.v, axis=-1), axes=[-1, 0]), axis=-1)
        else:
            raise ValueError("attention_type must be scaled_dot_product or additive")

        key_masks = tf.tile(key_masks, [self.head_num, 1])

        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

        # (h*N, T_q, T_k)

        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )
        if self.blinding:
            try:
                outputs = tf.matrix_set_diag(outputs, tf.ones_like(outputs)[
                                                      :, :, 0] * (-2 ** 32 + 1))
            except AttributeError:
                outputs = tf.compat.v1.matrix_set_diag(outputs, tf.ones_like(outputs)[
                                                                :, :, 0] * (-2 ** 32 + 1))

        outputs -= tf.reduce_max(outputs, axis=-1, keep_dims=True)
        outputs = softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # (h*N, T_q, T_k)
        query_masks = tf.tile(tf.expand_dims(
            query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)
        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
            result += queries
        if self.use_layer_norm:
            result = self.ln(result)

        if self.use_feed_forward:
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

        if self.output_type == "mean":
            return tf.reduce_mean(result, axis=1, keep_dims=True)
        elif self.output_type == "sum":
            return tf.reduce_sum(result, axis=1, keep_dims=True)
        else:
            return result

    def compute_output_shape(self, input_shape):

        return (None, 1, self.att_embedding_size * self.head_num)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num,
                  'dropout_rate': self.dropout_rate, 'use_res': self.use_res,
                  'use_positional_encoding': self.use_positional_encoding, 'use_feed_forward': self.use_feed_forward,
                  'use_layer_norm': self.use_layer_norm, 'seed': self.seed, 'supports_masking': self.supports_masking,
                  'blinding': self.blinding, 'attention_type': self.attention_type, 'output_type': self.output_type}
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class LayerNormalization(Layer):
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
class PositionEncoding(Layer):
    def __init__(self, pos_embedding_trainable=True,
                 zero_pad=False,
                 scale=True, **kwargs):
        self.pos_embedding_trainable = pos_embedding_trainable
        self.zero_pad = zero_pad
        self.scale = scale
        super(PositionEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        _, T, num_units = input_shape.as_list()  # inputs.get_shape().as_list()
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * (i // 2) / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        if self.zero_pad:
            position_enc[0, :] = np.zeros(num_units)
        self.lookup_table = self.add_weight("lookup_table", (T, num_units),
                                            initializer=tf.initializers.identity(position_enc),
                                            trainable=self.pos_embedding_trainable)

        # Be sure to call this somewhere!
        super(PositionEncoding, self).build(input_shape)

    def call(self, inputs, mask=None):
        _, T, num_units = inputs.get_shape().as_list()
        position_ind = tf.expand_dims(tf.range(T), 0)
        outputs = tf.nn.embedding_lookup(self.lookup_table, position_ind)
        if self.scale:
            outputs = outputs * num_units ** 0.5
        return outputs + inputs

    def compute_output_shape(self, input_shape):

        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self, ):

        config = {'pos_embedding_trainable': self.pos_embedding_trainable, 'zero_pad': self.zero_pad,
                  'scale': self.scale}
        base_config = super(PositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionSequencePoolingLayer(Layer):
    """The Attentional sequence pooling operation used in DIN.

      Input shape
        - A list of three tensor: [query,keys,keys_length]

        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

        - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **att_activation**: Activation function to use in attention net.

        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

        - **supports_masking**:If True,the input need to support masking.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            if not isinstance(input_shape, list) or len(input_shape) != 3:
                raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
                                 'on a list of 3 inputs')

            if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
                raise ValueError(
                    "Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
                        len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))

            if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
                raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
                                 'inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
                                 'Got different shapes: %s' % (input_shape))
        else:
            pass
        self.local_att = LocalActivationUnit(
            self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, )
        super(AttentionSequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            queries, keys = inputs
            key_masks = tf.expand_dims(mask[-1], axis=1)

        else:

            queries, keys, keys_length = inputs
            hist_len = keys.get_shape()[1]
            key_masks = tf.sequence_mask(keys_length, hist_len)

        attention_score = self.local_att([queries, keys], training=training)

        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        if tf.__version__ < '1.13.0':
            outputs._uses_learning_phase = attention_score._uses_learning_phase
        else:
            outputs._uses_learning_phase = training is not None

        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[1][1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):

        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

      Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape[0], input_shape[1]))
        size = 4 * \
            int(input_shape[0][-1]
                ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")
        self.dnn = DNN(self.hidden_units, self.activation, self.l2_reg, self.dropout_rate, self.use_bn, seed=self.seed)

        super(LocalActivationUnit, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = self.dnn(att_input, training=training)

        attention_score = tf.nn.bias_add(tf.tensordot(att_out, self.kernel, axes=(-1, 0)), self.bias)

        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(LocalActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Dice(Layer):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

      Input shape
        - Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

      Output shape
        - Same shape as the input.

      Arguments
        - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).

        - **epsilon** : Small float added to variance to avoid dividing by zero.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        # tf.layers.batch_normalization(
        # inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

try:
    unicode
except NameError:
    unicode = str
def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, str)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))