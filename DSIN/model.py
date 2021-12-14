
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Flatten, Input)


import tensorflow as tf
from layer import Embedding_layer,BiasEncoding,Muti_Attention,Din_attention,Dense_layer,Din_attention2


# 模型静态图构建和送入数据时的维度不同导致

class DSIN(tf.keras.Model):
    def __init__(self,fd,sess_feature_list, embedding_size=4, sess_max_count=5, sess_len_max=10,
                 att_embedding_size=2, att_head_num=4, dnn_hidden_units=(200, 80),spare_list =None):

        '''
        代码首先进行数值判断，要保证 Multi-Head Self-Attention 的输出结果 embedding 的维度和输入是相同的。
        首先，每个稀疏特征都会被映射为大小等于 embedding_size 的向量，而一个行为使用 (cate_id, brand) 来表示，
        该行为对应的 embedding 大小是这两个稀疏特征对应的 embedding 向量进行 concatenation 得到的，
        即行为的 embedding 大小为 2 * embedding_size (len(sess_feature_list) == 2)
        '''
        if (att_embedding_size * att_head_num != 2 * embedding_size):
            raise ValueError(
                "len(session_feature_lsit) * embedding_size must equal to att_embedding_size * att_head_num ,got %d * %d != %d *%d" % (
                    len(self.sess_feature_list), embedding_size, att_embedding_size, att_head_num))

        super(DSIN,self).__init__()
        self.fd = fd
        self.sess_feature_list = sess_feature_list
        self.sess_max_count=sess_max_count
        self.embedding_size = embedding_size

        # embedding layer
        self.embedding_layer = Embedding_layer(self.fd["sparse"],embedding_size,self.sess_feature_list)
        # pos embedding layer
        self.pos_embedding = BiasEncoding(self.sess_max_count)
        # Multi head attention layer
        # DSIN在计算每个sess内部的attention时，是没有用到mask的，我理解是因为2点，1,每个sess内的行为长度为10，要获得的输出是代表这个
        # sess整体的，如果行为是被padding到10 的 那代表用户的行为少，这也是一个信息。如果进行了mask，那用户行为长度这个信息就回被丢失
        # deepctr 这里supports_masking=True，感觉会产生迷惑。
        self.mul_att = Muti_Attention(att_embedding_size,att_head_num,use_positional_encoding=False,supports_masking=False)
        # din attention layer

        self.attention = Din_attention(self.sess_max_count,stag="frist")
        self.attention_2 = Din_attention(self.sess_max_count,stag="second3")

        # Bi-lstm
        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=2*embedding_size, state_is_tuple=True)

        # DNN
        self.dense_layer = Dense_layer(dnn_hidden_units)

    def call(self,input_data):
        sparse_input, dense_input, user_behavior_input_dict, user_sess_length = input_data[0:15],input_data[15],input_data[16:26],input_data[26]
        print("train",dense_input)
        #print("********************",sparse_input,len(sparse_input))
        #embedding_input = {"sparse_fg_list":self.fd["sparse"],"sess_feature_list":self.sess_feature_list}
        flag = 0
        # 给call函数传参 除了input，其他变量要用默认值。
        # target item 的embedding
        # 目标商品的embedding
        query_emb_list = self.embedding_layer(sparse_input,flag=flag)
        #print(query_emb_list)
        # 这里用tf.concat 就报错？？？？？
        query_emb = tf.keras.layers.concatenate(query_emb_list,axis=-1,name="7070707")
        query_emb = tf.expand_dims(query_emb, 1)
        print("que",query_emb)
        # dnn输入的embedding
        flag = 1
        deep_input_emb_list = self.embedding_layer(sparse_input,flag=flag)
        print("deep",deep_input_emb_list)
        # 这里用tf.concat 就报错？？？？？
        deep_input_emb = tf.keras.layers.concatenate(deep_input_emb_list,axis=-1,name="7979797")
        #deep_input_emb = tf.concat(deep_input_emb_list, axis=-1, name="7979797")
        deep_input_emb = Flatten()(deep_input_emb)
        #print(deep_input_emb)  #[B,60]
        """
        tr_input: list, 长度等于 sess_max_count=5, 每个元素为 [B, 10, 8] 大小的 Tensor, 
        10 表示 max_session_len, tr_input 的前缀 tr_ 表示 transformer，说明该变量是 Transformer 的输入
        每个用户有10个行为，每个行为有一个拼接后的dim=8的embedding
        """
        flag = 2
        tr_input = self.embedding_layer(user_behavior_input_dict,flag=flag,sess_max_count=self.sess_max_count)
        #print(len(tr_input),tr_input[0].shape)
        # 加入bias
        tr_input = self.pos_embedding(tr_input)
        #print(len(tr_input),tr_input[0].shape)

        # 进行multi head att
       # print("user_sess_length", user_sess_length)
        tr_out = []
        for i in range(self.sess_max_count):
            tr_out.append(self.mul_att([tr_input[i], tr_input[i]]))
        tr_out = tf.keras.layers.concatenate(tr_out,axis=1,name="fisstttt")
        print("trout",tr_out) #(B,5,8)
        #tr_out = tf.split(tr_out,8,axis=0)[0]
        # 进行attention 送入 target item的embedding以及 mask
        tf.print(tr_out)
        tf.print(query_emb)
        att_out,att_score = self.attention(tr_out,query=query_emb,mask=user_sess_length)

       # print(att_out.shape,att_score.shape) #(?, 1, 8) (?, 1, 5)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell, cell_bw=self.cell, dtype=tf.float32, inputs=tr_out
        )
        # outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。
        # output_state_fw和output_state_bw的类型为LSTMStateTuple，由（c,h）组成，分别代表memory cell 和hidden state.
        bi_lstm_out = tf.add(outputs[0], outputs[1])
        #print("bi_lstm_out",bi_lstm_out) #(?, 5, 8)
        #bi_lstm_out = tf.split(bi_lstm_out,8,axis=0)[0]
        # bi-lstm 的结果也送入attention
        lstm_attention_out,lstm_attention_score = self.attention_2(bi_lstm_out,query=query_emb,mask=user_sess_length)
        #(?, 1, 8)
        # 进行送入DNN前的concat
        #lstm_attention_out = tf.reduce_sum(bi_lstm_out,axis=1)
        print("llout",lstm_attention_out)
        query_emb = tf.squeeze(query_emb,1)
        att_out = tf.squeeze(att_out,1)
        lstm_attention_out = tf.squeeze(lstm_attention_out,1)
        # 静态图时会增加维度，但是送入数据时不会，所以必须要reshape
        dense_input = tf.expand_dims(dense_input,1)
        dense_input = tf.reshape(dense_input,[-1,1])

        dnn_in_1 = tf.concat([att_out,lstm_attention_out,deep_input_emb,dense_input],axis=-1,name="ccc1111")

        #dnn_in = tf.reshape(dnn_in_1,[-1,77])
        dnn_out = self.dense_layer(dnn_in_1)

        return dnn_out
