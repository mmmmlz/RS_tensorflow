import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense, Dropout
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.initializers import TruncatedNormal

from tensorflow.python.keras.layers import Embedding
from tools import prelu,dice,prelu2
from config import FEAD_DICT



class Embedding_layer(Layer):
    def __init__(self, fd,embedding_size,sess_feature):
        super(Embedding_layer, self).__init__()
        self.fd = fd  # 特征总数
        self.embedding_size = embedding_size
        print("1111",fd)

        self.sess_feature = sess_feature
        self.embedding_dict = {}
        #print(self.name)
    def build(self,input_shape):
        #super(Embedding_layer, self).build(input_shape)
        for i,feed in enumerate(self.fd):
            emd = Embedding(feed.vocabulary_size, self.embedding_size,name='sparse_emb_' +str(i) + '-' + feed.name,
                             embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=1024),
                             embeddings_regularizer=l2(1e-6),
                             # 布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。
                             # 设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。
                             # 如果该值为True，则下标0在字典中不可用，input_dim应设置为 | vocabulary | + 1
                             mask_zero=( feed.name in self.sess_feature)
                             )
            self.embedding_dict[feed.name] = emd
        print("embedding_dict",self.embedding_dict)

    def call(self,input_dict,flag=0,sess_max_count=5,**kwargs):
        #print(self.embedding_dict)
        print("calll",flag,"  length::",len(input_dict))
        embedding_vec_list = []

        # target item embedding
        if flag == 0:
            # for fg in self.fd:
            #     feat_name = fg.name
            #     if feat_name in self.sess_feature:
            #         #print(fg.hash_flag)
            #         lookup_idx = input_dict[feat_name]
            #             #print(lookup_idx)
            #         #print(self.embedding_dict[feat_name](lookup_idx))
            #         embedding_vec_list.append(self.embedding_dict[feat_name](lookup_idx))
            for feed in self.sess_feature:
                lookup_idx = FEAD_DICT.index(feed)
                embedding_vec_list.append(self.embedding_dict[feed](input_dict[lookup_idx]))


        if flag ==1:
            # for fg in self.fd:
            #     feat_name = fg.name
            #     lookup_idx = input_dict[feat_name]
            #     embedding_vec_list.append(self.embedding_dict[feat_name](lookup_idx))
            for feed in FEAD_DICT:

                lookup_idx = FEAD_DICT.index(feed)
                embedding_vec_list.append(self.embedding_dict[feed](input_dict[lookup_idx]))

        if flag ==2:
            print("input",input_dict)
            print(type(input_dict))
            # 这部分是对用户历史行为进行embedding，对每一个sess，所有用户依次取查询其cate_id 和brand的embedding 组合起来返回
            # for i in range(sess_max_count):
            #     sess_name = "sess_" + str(i)
            #     #temp =input_dict[sess_name]
            #     k = []
            #     for fg in self.sess_feature:
            #         feat_name = fg
            #         #print(input_dict[sess_name])
            #         lookup_idx = input_dict[sess_name][feat_name]
            #         k.append(self.embedding_dict[feat_name](lookup_idx))
            #
            #     tmp = tf.concat(k,axis=-1)
            #     embedding_vec_list.append(tmp)
            for i in range(0,len(input_dict),2):
                cate = self.embedding_dict["cate_id"](input_dict[i])
                brand = self.embedding_dict["brand"](input_dict[i+1])
                print(cate.shape,brand.shape)
                tmp = tf.concat([cate,brand],axis=-1)
                embedding_vec_list.append(tmp)

        print(len(embedding_vec_list))
        return embedding_vec_list


class BiasEncoding(Layer):
    def __init__(self, sess_max_count, seed=1024, **kwargs):
        self.sess_max_count = sess_max_count
        self.seed = seed
        super(BiasEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print("inputshape",input_shape)
        if self.sess_max_count == 1:
            embed_size = input_shape[2].value
            seq_len_max = input_shape[1].value
        else:
            embed_size = input_shape[0][2].value
            seq_len_max = input_shape[0][1].value

        self.sess_bias_embedding = self.add_weight('sess_bias_embedding', shape=(self.sess_max_count, 1, 1),
                                                   initializer=TruncatedNormal(
                                                       mean=0.0, stddev=0.0001, seed=self.seed),trainable=True)
        self.seq_bias_embedding = self.add_weight('seq_bias_embedding', shape=(1, seq_len_max, 1),
                                                  initializer=TruncatedNormal(
                                                      mean=0.0, stddev=0.0001, seed=self.seed),trainable=True)
        self.item_bias_embedding = self.add_weight('item_bias_embedding', shape=(1, 1, embed_size),
                                                   initializer=TruncatedNormal(
                                                       mean=0.0, stddev=0.0001, seed=self.seed),trainable=True)

        # Be sure to call this somewhere!
        super(BiasEncoding, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        transformer_out = []
        for i in range(self.sess_max_count):
            transformer_out.append(
                #[B,10,8] + [1,1,8]+[1,10,1]+[1,1] 由广播机制可以相加
                inputs[i] + self.item_bias_embedding + self.seq_bias_embedding + self.sess_bias_embedding[i])

        return transformer_out


class Muti_Attention(Layer):
    """  Simplified version of Transformer  proposed in 《Attention is all you need》

      Input shape
        - a list of two 3D tensor with shape ``(batch_size, timesteps, input_dim)`` if supports_masking=True.
        - a list of two 4 tensors, first two tensors with shape ``(batch_size, timesteps, input_dim)``,last two tensors with shape ``(batch_size, 1)`` if supports_masking=False.
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, input_dim)``.
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

      References
            - [Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
    """

    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=False, blinding=True, seed=1024, supports_masking=False,
                 **kwargs):
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
        super(Muti_Attention, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):

        embedding_size = input_shape[0][-1].value
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (self.att_embedding_size,self.head_num,embedding_size))
        self.seq_len_max = input_shape[0][-2].value
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
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
        self.ln = tf.contrib.layers.layer_norm
        # Be sure to call this somewhere!
        super(Muti_Attention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            queries, keys = inputs
            print(queries.shape)
            query_masks, key_masks = mask
            #     query_masks,key_masks = mask,mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            #queries, keys, query_masks, key_masks = inputs

            queries, keys = inputs

        if self.use_positional_encoding:
            queries = positional_encoding(queries)
            keys = positional_encoding(queries)

        #     [B,10,8]X[8,8] = [B,10,8]
        querys = tf.tensordot(queries, self.W_Query,
                              axes=(-1, 0))  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))

        # head_num*None T_q D
        # 将querys 分成8个[B,10,1] 之后在第一个维度拼接 [8*B,10,1]
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        # head_num*None T_q T_k
        # 计算每一个行为之间的att score [8*B,10,1]*[8*B,1,10] = [8*B,10,10]
        outputs = tf.matmul(querys, keys, transpose_b=True)

             # 归一化
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

             # 对每一个[10,10]的矩阵进行对角化直为一个最小值，表示自己和自己的att score = 0
        if self.blinding:
            outputs = tf.matrix_set_diag(outputs, tf.ones_like(outputs)[:, :, 0] * (-2 ** 32 + 1))
            # 在最后一维求max [8*B,10,1] [8*B,10,10]-[8*B,10,1]
        outputs -= tf.reduce_max(outputs, axis=-1, keep_dims=True)
            # 将输出变为softmax概率
            # print("1:",outputs.shape)
        outputs = tf.nn.softmax(outputs)
            # print("2:",outputs.shape)
        outputs = self.dropout(outputs, training=training)
            # Weighted sum
            # ( h*N, T_q, C/h)
            # [8*B,10,10] [8*B,10,1]
            # print("3:",outputs.shape,values.shape)
        result = tf.matmul(outputs, values)
            #[8*B,10,1]
            # 采用split将其分解为8个[B,10,1]在第二维拼接 得到[B,10,8]
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
        # 结果在第一维，也就是求一个sess内10个行为的均值，作为这个sess的表示
        return tf.reduce_mean(result, axis=1, keep_dims=True)


class Din_attention(Layer):
    def __init__(self,sess_max_count,stag):
        super(Din_attention,self).__init__()
        self.sess_max_count = sess_max_count
        self.stag = stag
    def build(self, input_shape):
        pass


    def call(self, inputs,query=None,mask=None, **kwargs):

        #变换维度做一层mlp
        query = tf.reshape(query,[-1,8])
        facts_size = inputs.get_shape().as_list()[-1]  # Hidden size for rnn layer

        query = tf.layers.dense(query, facts_size, activation=None, name='f11' + self.stag)
        query = tf.reshape(query, [-1,1,8])
        # 1. 转换query维度，变成历史维度T
        # query是[B, H]，转换到 queries 维度为(B, T, H)，为了让pos_item和用户行为序列中每个元素计算权重
        # 此时query是 Tensor("concat:0", shape=(?, 36), dtype=float32)
        # tf.shape(keys)[1] 结果就是 T，query是[B, H]，经过tile，就是把第一维按照 T 展开，得到[B, T * H]
        #queries = tf.tile(query, [1,5],name="3214")  # Batch * Time * Hidden size
        queries = tf.keras.layers.concatenate([query for _ in range(int(inputs.shape[1]))],axis=1)

        queries = tf.reshape(queries, [-1,int(inputs.shape[1]),8])

        # 2. 这部分目的就是为了在MLP之前多做一些捕获行为item和候选item之间关系的操作：加减乘除等。
        # 得到 Local Activation Unit 的输入。即 候选广告 queries 对应的 emb，用户历史行为序列 facts
        # 对应的 embed, 再加上它们之间的交叉特征, 进行 concat 后的结果
        din_all = tf.keras.layers.concatenate([queries, inputs, queries - inputs,queries * inputs],
                            axis=-1)  # Batch * Time * (4 * Hidden size)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att'+self.stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att'+self.stag )
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att'+self.stag)  # Batch * Time * 1

        # 上一层 d_layer_3_all 的 shape 为 [B, T, 1]
        # 下一步 reshape 为 [B, 1, T], axis=2 这一维表示 T 个用户行为序列分别对应的权重参数
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(inputs)[1]])  # Batch * 1 * time

        scores = d_layer_3_all  # attention的输出, [B, 1, T]
        print("s",scores.shape)

        # 进行mask 这里的mask表示每个用户的sess数量，对于[B,1] 中每一个数A，与self.sess_max_count 得到[True,True......false] 长度为self.sess_max_count True的个数为A
        key_masks = tf.sequence_mask( mask, self.sess_max_count, dtype=tf.float32) # Batch * 1 * Time
        #if key_masks.shape[0] is None:

        key_masks = tf.expand_dims(key_masks,1)
        key_masks = tf.reshape(key_masks,[-1,1,5])
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        # 使用tf.equal将其变为bool
        t = tf.equal(key_masks, 1)
        scores = tf.where(t, scores, paddings,name="sqqqwhrer")  # [B, 1, T] ，没有的地方用paddings填充
        scores = tf.nn.softmax(scores)  # [B, 1, T]
        #print(scores.shape)
        output = tf.matmul(scores,inputs)

        return output,scores

class Din_attention2(Layer):
    def __init__(self,sess_max_count,stag):
        super(Din_attention2,self).__init__()
        self.sess_max_count = sess_max_count
        self.stag = stag
    def build(self, input_shape):
        pass


    def call(self, inputs,query=None,mask=None, **kwargs):


        facts_size = inputs.get_shape().as_list()[-1]  # Hidden size for rnn layer
        #print("ttttttttt",inputs.shape,query.shape)
       # inputs = tf.split(inputs, 8, axis=0)[0]
        query = tf.layers.dense(query, facts_size, activation=None, name='f11'+self.stag)
        print("qqqqqqe",query)
        # output = tf.reduce_sum(inputs,axis=1)
        # scores = output
        # if self.stag=="second":
        #     query = prelu2(query, scope=self.stag)
        # else:
        #     query = prelu(query,scope=self.stag)
        query = tf.nn.relu(query)
        query = tf.squeeze(query,axis=1)
        print("qqqqqqe", query)
        # 1. 转换query维度，变成历史维度T
        # query是[B, H]，转换到 queries 维度为(B, T, H)，为了让pos_item和用户行为序列中每个元素计算权重
        # 此时query是 Tensor("concat:0", shape=(?, 36), dtype=float32)
        # tf.shape(keys)[1] 结果就是 T，query是[B, H]，经过tile，就是把第一维按照 T 展开，得到[B, T * H]
        queries = tf.tile(query, [1, tf.shape(inputs)[1]])  # Batch * Time * Hidden size
        print('s213123',queries)
        queries = tf.reshape(queries, [-1,5,8])
        print("rrrrrrr",queries)

        # 2. 这部分目的就是为了在MLP之前多做一些捕获行为item和候选item之间关系的操作：加减乘除等。
        # 得到 Local Activation Unit 的输入。即 候选广告 queries 对应的 emb，用户历史行为序列 facts
        # 对应的 embed, 再加上它们之间的交叉特征, 进行 concat 后的结果
        din_all = tf.concat([queries, inputs, queries - inputs,queries - inputs],
                            axis=-1)  # Batch * Time * (4 * Hidden size)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att'+self.stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att'+self.stag )
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att'+self.stag)  # Batch * Time * 1

        # 上一层 d_layer_3_all 的 shape 为 [B, T, 1]
        # 下一步 reshape 为 [B, 1, T], axis=2 这一维表示 T 个用户行为序列分别对应的权重参数
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(inputs)[1]])  # Batch * 1 * time

        scores = d_layer_3_all  # attention的输出, [B, 1, T]
        print("s",scores.shape)

        # 进行mask 这里的mask表示每个用户的sess数量，对于[B,1] 中每一个数A，与self.sess_max_count 得到[True,True......false] 长度为self.sess_max_count True的个数为A
        key_masks = tf.sequence_mask(mask, self.sess_max_count, dtype=tf.float32) # Batch * 1 * Time
        key_masks = tf.expand_dims(key_masks,1)

        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        # 使用tf.equal将其变为bool
        scores = tf.where(tf.equal(key_masks, 1), scores, paddings)  # [B, 1, T] ，没有的地方用paddings填充

        scores = tf.nn.softmax(scores)  # [B, 1, T]

        #print(scores.shape)
        output = tf.matmul(scores,inputs)

        return output,scores

class Dense_layer(Layer):
    def __init__(self, hidden_units, out_dim=1, activation='relu', dropout=0.0):
        super(Dense_layer, self).__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(out_dim, activation=None)
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        # inputs: [None, n*k]
        print(self.hidden_layers)
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.dropout(x)
        output = self.out_layer(x)
        return output