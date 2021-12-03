from layers import Linear, Dense_layer, CIN

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, Embedding
import numpy as np
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Dense


class XdeepFM(BaseEstimator,TransformerMixin):
#BaseEstimator, TransformerMixin
    def __init__(self, cate_feature_size, field_size,numeric_feature_size,
                 embedding_size=8,
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True,cin_layer=[124,124]):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.cate_feature_size = cate_feature_size
        self.numeric_feature_size = numeric_feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.total_size = self.field_size * self.embedding_size + self.numeric_feature_size
        self.deep_layers = deep_layers
        self.cin_layer = cin_layer
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result,self.valid_result = [],[]

        self.cross_layer_num = len(self.cin_layer)
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feat_index = tf.placeholder(tf.int32,
                                             shape=[None,None],
                                             name='feat_index')
            self.feat_value = tf.placeholder(tf.float32,
                                           shape=[None,None],
                                           name='feat_value')

            self.numeric_value = tf.placeholder(tf.float32,[None,None],name='num_value')

            self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool,name='train_phase')

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
            self.embeddings = tf.multiply(self.embeddings,feat_value)

            self.x0 = tf.concat([self.numeric_value,
                                 tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])]
                                ,axis=1)

            #

            # deep part

            self.y_deep = tf.nn.dropout(self.x0,self.dropout_keep_deep[0])

            for i in range(0,len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["deep_layer_%d" %i]), self.weights["deep_bias_%d"%i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])

            print("xo",self.x0.shape)
            self.x1 = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])

            self.out_liner = self.liner(self.x1)

            # cross_part
            cin_in = tf.reshape(self.embeddings, (-1,self.field_size,self.embedding_size))
            # cin_in [B,30,8]
            dim = int(self.embedding_size)
            hidden_nn_layers = [cin_in]
            final_result = []
            split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
            #print(self.field_nums)
            for idx, layer_size in enumerate(self.cin_layer):
                split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
                #[B,30,1]
                dot_result_m = tf.matmul(
                    split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(
                    dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
                curr_out = tf.nn.conv1d(
                    dot_result, filters=self.filters[idx], stride=1, padding='VALID')
                curr_out = tf.nn.bias_add(curr_out, self.bias[idx])
                #curr_out = self.activation_layers[idx](curr_out)
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
                final_result.append(curr_out)
                hidden_nn_layers.append(curr_out)
            final_result = final_result[1:]  # 去掉X0
            print(final_result)
            res = tf.concat(final_result, axis=1)  # (None, field_num[1]+...+field_num[n], k)
            output = tf.reduce_sum(res, axis=-1)  # (None, field_num[1]+...+field_num[n])
            # concat_part
            print(self.out_liner.shape)
            concat_input = tf.concat([output, self.y_deep], axis=1)
            concat_input = tf.add(concat_input,self.out_liner)
            print(concat_input.shape)
            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["deep_layer_%d" % i])
                for i in range(self.cross_layer_num):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.filters[i])


            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)


            #init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)





    def _initialize_weights(self):
        weights = dict()

        #embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01),
            name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.cate_feature_size,1],0.0,1.0),name='feature_bias')

        self.liner = Dense(1, activation=None)
        #deep layers
        num_layer = len(self.deep_layers)
        glorot = np.sqrt(2.0/(self.total_size + self.deep_layers[0]))

        weights['deep_layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(self.total_size,self.deep_layers[0])),dtype=np.float32
        )
        weights['deep_bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32
        )


        for i in range(1,num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["deep_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["deep_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # CIN layer

        self.field_nums = [self.field_size]
        self.filters = []
        self.bias = []
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        for i, size in enumerate(self.cin_layer):
            self.filters.append(tf.Variable(initial_value=w_init(shape=(1, self.field_nums[-1] * self.field_nums[0], size), dtype=np.float32),
            trainable=True,))

            self.bias.append(tf.Variable(initial_value=b_init(shape=(size),dtype=np.float32)))
            self.field_nums.append(size)


        # final concat projection layer


        input_size = self.total_size + self.deep_layers[-1]

        glorot = np.sqrt(2.0/(input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=((len(self.cin_layer)-1)*self.cin_layer[-1]+self.deep_layers[-1],1)),dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)


        return weights


    def get_batch(self,Xi,Xv,Xv2,y,batch_size,index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end],Xv[start:end],Xv2[start:end],[[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c,d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    def predict(self, Xi, Xv,Xv2,y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y

        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.numeric_value: Xv2,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase: True}

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss


    def fit_on_batch(self,Xi,Xv,Xv2,y):

        feed_dict = {self.feat_index:Xi,
                     self.feat_value:Xv,
                     self.numeric_value:Xv2,
                     self.label:y,
                     self.dropout_keep_deep:self.dropout_dep,
                     self.train_phase:True}

        loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)

        return loss

    def fit(self, cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train,
            cate_Xi_valid=None, cate_Xv_valid=None, numeric_Xv_valid=None,y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        print(len(cate_Xi_train))
        print(len(cate_Xv_train))
        print(len(numeric_Xv_train))
        print(len(y_train))
        has_valid = cate_Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch = self.get_batch(cate_Xi_train, cate_Xv_train, numeric_Xv_train,y_train, self.batch_size, i)

                self.fit_on_batch(cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch)


            if has_valid:
                y_valid = np.array(y_valid).reshape((-1,1))
                loss = self.predict(cate_Xi_valid, cate_Xv_valid, numeric_Xv_valid, y_valid)
                print("epoch",epoch,"loss",loss)









