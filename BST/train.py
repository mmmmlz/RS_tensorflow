# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/28  9:31 
# @file   ：train.PY
# @Tool   ：PyCharm


from tools import *
from model import BST


def check_model(model, model_name, x, y, check_model_io=True):
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, batch_size=100, epochs=2, validation_split=0.5)



if __name__ == '__main__':
    model_name = "BST"
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = BST(dnn_feature_columns=feature_columns,
                history_feature_list=behavior_feature_list,
                att_head_num=4)
    check_model(model, model_name, x, y,
                check_model_io=True)
