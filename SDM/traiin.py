# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/15  15:39 
# @file   ：traiin.py.PY
# @Tool   ：PyCharm

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tools import get_xy_fd_sdm,sampledsoftmaxloss
from model import SDM

def check_model(model,x,y):
    model.fit(x, y, batch_size=10, epochs=2, validation_split=0.5)
if __name__ == '__main__':
    x,y,user_feature_columns,item_feature_columns, history_feature_list = get_xy_fd_sdm()
    model = SDM(user_feature_columns,item_feature_columns,history_feature_list,units=8)
    model.compile('adam', sampledsoftmaxloss)
    check_model(model, x, y)

