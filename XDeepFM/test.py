import tensorflow as tf

import pandas as pd
import numpy as np
from tensorflow.keras import losses, optimizers
import config

from sklearn.model_selection import StratifiedKFold
from DataLoader import FeatureDictionary, DataParser

from XdeepFM import XdeepFm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test,


def run_base_model_dcn(dfTrain, dfTest, folds, XdeepFM_params):

    fd = FeatureDictionary(dfTrain,dfTest,numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           cate_cols = config.CATEGORICAL_COLS)

    print(fd.feat_dim)
    print(fd.feat_dict)

    data_parser = DataParser(feat_dict=fd)
    cate_Xi_train, cate_Xv_train, numeric_Xv_train,y_train = data_parser.parse(df=dfTrain, has_label=True)
    cate_Xi_test, cate_Xv_test, numeric_Xv_test,ids_test = data_parser.parse(df=dfTest)

    XdeepFM_params["cate_feature_size"] = fd.feat_dim
    XdeepFM_params["field_size"] = len(cate_Xi_train[0])
    XdeepFM_params['numeric_feature_size'] = len(config.NUMERIC_COLS)

    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        cate_Xi_train_, cate_Xv_train_, numeric_Xv_train_,y_train_ = _get(cate_Xi_train, train_idx), _get(cate_Xv_train, train_idx),_get(numeric_Xv_train, train_idx), _get(y_train, train_idx)
        cate_Xi_valid_, cate_Xv_valid_, numeric_Xv_valid_,y_valid_ = _get(cate_Xi_train, valid_idx), _get(cate_Xv_train, valid_idx),_get(numeric_Xv_train, valid_idx), _get(y_train, valid_idx)

        Xdeepfm = XdeepFm(**XdeepFM_params)
        Xdeepfm.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy'], )
        cate_Xi_train_ = np.array(cate_Xi_train_,dtype="float32")
        cate_Xv_train_ = np.array(cate_Xv_train_,dtype="float32")
        numeric_Xv_train_ = np.array(numeric_Xv_train_,dtype="float32")
        y_train_ = np.array(y_train_)
        optimizer = optimizers.SGD(0.01)
        x_train = {"cate_idx":cate_Xi_train_,
                "cate_value":cate_Xv_train_,
                   "numeric":numeric_Xv_train_
                   }


        Xdeepfm.fit(x_train,y_train_)
        pre = Xdeepfm(x_train)
        print(pre)
        # with tf.GradientTape() as tape:
        #     y_pre = Xdeepfm(x_train)
        #     loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
        #     grad = tape.gradient(loss, Xdeepfm.variables)
        #     optimizer.apply_gradients(grads_and_vars=zip(grad, Xdeepfm.variables))
        break


dfTrain, dfTest, X_train, y_train, X_test, ids_test = load_data()
print('load_data_over')
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))
print('process_data_over')

XdeepFM_params = {

    "embedding_size": 8,
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,


    "cin_layer":[124,124]
}
print('start train')
run_base_model_dcn(dfTrain, dfTest, folds, XdeepFM_params)
