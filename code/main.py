# -*- coding: utf-8 -*-#
'''
# Name:         n麦
# Description:  
# Author:       neu
# Date:         2020/7/28
'''
from os.path import join as pjoin

from data_loader.data_utils import *
from utils.math_graph import *

from keras.layers import Input, Dropout, Conv2D, add
from keras.models import Model, Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from keras.regularizers import l2

from models.layer import *

n = 228
n_his = 12
n_pred = 9

W = weight_matrix(pjoin('./data_loader/data', f'W_228.csv'))
data_file = f'V_228.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./data_loader/data', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

x = PeMS.get_data('train')
x = x.transpose(0,3,1,2)
print(x.shape)

x_train = x[:1000]
y_train = x[1:1001]

print(x_train.shape)
print(y_train.shape)

def build_model():
    inputs = Input(shape=(1,21,228))
    layer1_1 = Conv2D(filters=228, kernel_size=(1, 3))(inputs)
    print(layer1_1)
    layer1_2 = Conv2D(filters=228, kernel_size=(1,3), activation='sigmoid')(inputs)
    print(layer1_2)
    layer2 = Conv2D(filters=228, kernel_size=(1,3))(inputs)
    print(layer2)
    out = add([layer1_1, layer1_2, layer2])
    print(out)
    out = Activation('relu')(out)
    print(out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = build_model()
model.fit(x_train, y_train,
          epochs=2)