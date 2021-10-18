'''
Model definitions
'''
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ReLU, LeakyReLU, \
    SpatialDropout2D, Dropout
from tensorflow_probability.python.layers import Convolution2DFlipout
from tensorflow.keras.models import Model

def tfBlockUnet(filters=3, transposed=False, kernel_size=3, bn=True, relu=True, pad="same", dropout=0., flipout=False,
                kdf=None, name='', spatial_dropout=True):
    block = Sequential(name=name)
    if relu:
        block.add(ReLU())
    else:
        block.add(LeakyReLU(0.2))
    if not transposed:
        block.add(Conv2D(filters=filters, kernel_size=kernel_size, padding=pad,
                         kernel_initializer=RandomNormal(0.0, 0.02), activation=None, strides=(2, 2)))
    else:
        block.add(UpSampling2D(interpolation='bilinear'))
        if flipout:
            block.add(Convolution2DFlipout(filters=filters, kernel_size=(kernel_size - 1), strides=(1, 1), padding=pad,
                                           data_format="channels_last", kernel_divergence_fn=kdf,
                                           activation=None))
        else:
            block.add(Conv2D(filters=filters, kernel_size=(kernel_size - 1), padding=pad,
                             kernel_initializer=RandomNormal(0.0, 0.02), strides=(1, 1), activation=None))

    if spatial_dropout:
        block.add(SpatialDropout2D(rate=dropout))
    else:
        block.add(Dropout(rate=dropout))

    if bn:
        block.add(BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9))

    return block


def Bayes_DfpNet(input_shape=(128, 128, 3), expo=5, dropout=0., flipout=False, kl_scaling=10000, spatial_dropout=True):
    channels = int(2 ** expo + 0.5)
    kdf = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(kl_scaling, dtype=tf.float32))

    layer1 = Sequential(name='layer1')
    layer1.add(Conv2D(filters=channels, kernel_size=4, strides=(2, 2), padding='same', activation=None,
                      data_format='channels_last'))
    layer2 = tfBlockUnet(filters=channels * 2, transposed=False, bn=True, relu=False, dropout=dropout, name='layer2', spatial_dropout=spatial_dropout)
    layer3 = tfBlockUnet(filters=channels * 2, transposed=False, bn=True, relu=False, dropout=dropout, name='layer3', spatial_dropout=spatial_dropout)
    layer4 = tfBlockUnet(filters=channels * 4, transposed=False, bn=True, relu=False, dropout=dropout, name='layer4', spatial_dropout=spatial_dropout)
    layer5 = tfBlockUnet(filters=channels * 8, transposed=False, bn=True, relu=False, dropout=dropout, name='layer5', spatial_dropout=spatial_dropout)
    layer6 = tfBlockUnet(filters=channels * 8, transposed=False, bn=True, relu=False, dropout=dropout, kernel_size=2,
                         pad='valid', name='layer6', spatial_dropout=spatial_dropout)
    layer7 = tfBlockUnet(filters=channels * 8, transposed=False, bn=True, relu=False, dropout=dropout, kernel_size=2,
                         pad='valid', name='layer7', spatial_dropout=spatial_dropout)

    # note, kernel size is internally reduced by one for the decoder part
    dlayer7 = tfBlockUnet(filters=channels * 8, transposed=True, bn=True, relu=True, dropout=dropout, flipout=flipout,
                          kdf=kdf, kernel_size=2, pad='valid', name='dlayer7', spatial_dropout=spatial_dropout)
    dlayer6 = tfBlockUnet(filters=channels * 8, transposed=True, bn=True, relu=True, dropout=dropout, flipout=flipout,
                          kdf=kdf, kernel_size=2, pad='valid', name='dlayer6', spatial_dropout=spatial_dropout)
    dlayer5 = tfBlockUnet(filters=channels * 4, transposed=True, bn=True, relu=True, dropout=dropout, flipout=flipout,
                          kdf=kdf, name='dlayer5', spatial_dropout=spatial_dropout)
    dlayer4 = tfBlockUnet(filters=channels * 2, transposed=True, bn=True, relu=True, dropout=dropout, flipout=flipout,
                          kdf=kdf, name='dlayer4', spatial_dropout=spatial_dropout)
    dlayer3 = tfBlockUnet(filters=channels * 2, transposed=True, bn=True, relu=True, dropout=dropout, flipout=flipout,
                          kdf=kdf, name='dlayer3', spatial_dropout=spatial_dropout)
    dlayer2 = tfBlockUnet(filters=channels, transposed=True, bn=True, relu=True, dropout=dropout, flipout=flipout,
                          kdf=kdf, name='dlayer2', spatial_dropout=spatial_dropout)
    dlayer1 = Sequential(name='outlayer')
    dlayer1.add(ReLU())
    dlayer1.add(Conv2DTranspose(3, kernel_size=4, strides=(2, 2), padding='same'))

    # forward pass
    inputs = Input(input_shape)
    out1 = layer1(inputs)
    out2 = layer2(out1)
    out3 = layer3(out2)
    out4 = layer4(out3)
    out5 = layer5(out4)
    out6 = layer6(out5)
    out7 = layer7(out6)
    # ... bottleneck ...
    dout6 = dlayer7(out7)
    dout6_out6 = tf.concat([dout6, out6], axis=3)
    dout6 = dlayer6(dout6_out6)
    dout6_out5 = tf.concat([dout6, out5], axis=3)
    dout5 = dlayer5(dout6_out5)
    dout5_out4 = tf.concat([dout5, out4], axis=3)
    dout4 = dlayer4(dout5_out4)
    dout4_out3 = tf.concat([dout4, out3], axis=3)
    dout3 = dlayer3(dout4_out3)
    dout3_out2 = tf.concat([dout3, out2], axis=3)
    dout2 = dlayer2(dout3_out2)
    dout2_out1 = tf.concat([dout2, out1], axis=3)
    dout1 = dlayer1(dout2_out1)

    return Model(inputs=inputs, outputs=dout1)


def block_bayesian_mars_moon(num_convs = 2, filters = 32, flipout=True, dropout=0., kdf = None, bn=True, spatial_dropout=True):
    ''' block of mars moon model
    :returns sequential model with convolutional layers
    '''
    block = Sequential()
    for i in range(num_convs):
        if i>0:
            block.add(LeakyReLU())
        if flipout:
            block.add(Convolution2DFlipout(filters=filters, kernel_size=5, padding="same", activation=None, kernel_divergence_fn=kdf))
        else:
            block.add(Conv2D(filters=filters, kernel_size=5, padding="same", activation=None))
    if spatial_dropout:
        block.add(SpatialDropout2D(rate=dropout))
    else:
        block.add(Dropout(rate=dropout))
    if bn:
        block.add(BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9))
    return block

def bayesian_mars_moon(in_shape, out_filters = 3, reg_filters = 32, flipout=True, dropout=0., kl_scaling=1, bn=True, spatial_dropout=True):

    # define kernel divergence function, will be used for every block
    kdf = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(kl_scaling, dtype=tf.float32))

    with tf.name_scope("bayesian_mars_moon") as scope:
        # blocks
        B0 = block_bayesian_mars_moon(num_convs=1, filters=reg_filters, flipout=flipout, dropout=dropout, bn = bn, kdf = kdf, spatial_dropout=spatial_dropout)
        L0 = block_bayesian_mars_moon(num_convs=2, filters=reg_filters, flipout=flipout, dropout=dropout, bn = bn, kdf = kdf, spatial_dropout=spatial_dropout)
        L1 = block_bayesian_mars_moon(num_convs=2, filters=reg_filters, flipout=flipout, dropout=dropout, bn = bn, kdf = kdf, spatial_dropout=spatial_dropout)
        L2 = block_bayesian_mars_moon(num_convs=2, filters=reg_filters, flipout=flipout, dropout=dropout, bn = bn, kdf = kdf, spatial_dropout=spatial_dropout)
        L3 = block_bayesian_mars_moon(num_convs=2, filters=reg_filters, flipout=flipout, dropout=dropout, bn = bn, kdf = kdf, spatial_dropout=spatial_dropout)
        L4 = block_bayesian_mars_moon(num_convs=2, filters=reg_filters, flipout=flipout, dropout=dropout, bn = bn, kdf = kdf, spatial_dropout=spatial_dropout)
        OUT = block_bayesian_mars_moon(num_convs=1, filters=out_filters, flipout=flipout, dropout=dropout, bn = bn, kdf = kdf, spatial_dropout=spatial_dropout)

        # forward pass
        input = Input(shape=in_shape)
        b0 = B0(input)
        b0 = tf.keras.layers.LeakyReLU()(b0)
        l0 = L0(b0)
        s0 = tf.keras.layers.add([b0, l0])
        b1 = tf.keras.layers.LeakyReLU()(s0)
        l1 = L1(b1)
        s1 = tf.keras.layers.add([b1, l1])
        b2 = tf.keras.layers.LeakyReLU()(s1)
        l2 = L2(b2)
        s2 = tf.keras.layers.add([b2, l2])
        b3 = tf.keras.layers.LeakyReLU()(s2)
        l3 = L3(b3)
        s3 = tf.keras.layers.add([b3, l3])
        b4 = tf.keras.layers.LeakyReLU()(s3)
        l4 = L4(b4)
        s4 = tf.keras.layers.add([b4, l4])
        b5 = tf.keras.layers.LeakyReLU()(s4)
        output = OUT(b5)

        return Model(inputs=input, outputs=output)
