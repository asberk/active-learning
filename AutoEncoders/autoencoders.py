from datetime import datetime
from keras import backend as K

from keras.layers import Input, Conv2D, Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Reshape, UpSampling2D
from keras import regularizers
from keras.callbacks import CSVLogger

from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist


def create_basic_encoder(input_shape, return_as_model=True,
                         name='basic_encoder'):
    input_img = Input(input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    if return_as_model:
        return Model(input_img, encoded, name=name)
    else:
        return (input_img, encoded)


def create_basic_decoder(input_shape, return_as_model=True,
                         name='basic_decoder'):
    input_enc = Input(shape=input_shape)
    y = Conv2D(8, (3, 3), activation='relu', padding='same')(input_enc)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(16, (3, 3), activation='relu')(y)
    y = UpSampling2D((2, 2))(y)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
    if return_as_model:
        return Model(input_enc, decoded, name=name)
    else:
        return (input_enc, decoded)


def create_autoencoder(encoder, decoder, name=None):
    return Model(encoder.inputs,
                 decoder(encoder(encoder.inputs)),
                 name=name)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return ((x_train, y_train), (x_test, y_test))


def simple_convolutional_autoencoder(input_shape=None,
                                     encoding_shape=None):
    # Make Encoder
    if input_shape is None:
        input_shape = (28, 28, 1)
    encoder = create_basic_encoder(input_shape,
                                   name='simple_convolutional_enc')
    # Make Decoder
    if encoding_shape is None:
        encoded_shape = encoder.output_shape[1:]
        # encoded_shape = (4, 4, 8)
    decoder = create_basic_decoder(encoded_shape,
                                   name='simple_convolutional_dec')
    # Make Autoencoder
    autoencoder = create_autoencoder(encoder, decoder,
                                     name='simple_convolutional_ae')
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return (autoencoder, encoder, decoder)


def regularized_convolutional_autoencoder(input_shape=None,
                                          encoded_size=None,
                                          regularizer=None):
    # Make Encoder
    if input_shape is None:
        input_shape = (28, 28, 1)
    if encoded_size is None:
        encoded_size = 10
    if regularizer is None:
        regularizer = regularizers.l1(1e-4)
    ae, e, d = simple_convolutional_autoencoder(input_shape)
    x = Flatten()(e.outputs)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(encoded_size, activation='sigmoid',
              activity_regularizer=regularizer)(x)
    encoded = Dropout(.5)(x)
    encoder = Model(e.inputs, encoded,
                    name='regularized_convolutional_enc')
    # Make Decoder
    d_input_shape = d.input_shape[1:]
    d_input_size = np.prod(d_input_shape)
    input_enc = Input(shape=encoder.output_shape[1:])
    y = Dense(64, activation='relu')(input_enc)
    y = Dropout(.5)(y)
    y = Dense(d_input_size, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Reshape(d_input_shape)(y)
    decoder_top = Model(input_enc, y)
    decoder = Model(decoder_top.inputs, d(decoder_top.outputs),
                    name='regularized_convolutional_dec')
    # Make Autoencoder
    autoencoder = create_autoencoder(encoder,
                                     decoder,
                                     name='regz_convolutional_ae')
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return (autoencoder, encoder, decoder)


def model_to_json(model, filepath):
    info = 'Writing {} to json file...'.format(model.name)
    print(info, end='')
    model_json = model.to_json()
    with open(filepath, 'w') as fp:
        fp.write(model_json)
    print('done!')
    return


def strftime():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    input_shape = x_train.shape[1:]
    scae, scenc, scdec = simple_convolutional_autoencoder(input_shape)
    print("training simple convolutional autoencoder")
    scae_logname = 'simple_convolutional_ae_' + strftime() + '.log'
    scae_logger = CSVLogger(scae_logname)

    scae.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True,
             validation_data=(x_test, x_test), callbacks=[scae_logger])
    model_to_json(scae, 'simple_convolutional_ae.json')
    model_to_json(scenc, 'simple_convolutional_enc.json')
    model_to_json(scdec, 'simple_convolutional_dec.json')

    rcae, rcenc, rcdec = regularized_convolutional_autoencoder(input_shape)
    rcae_logname = 'regularized_convolutional_ae_' + strftime() + '.log'
    rcae_logger = CSVLogger(rcae_logname)
    print('Training regularized convolutional autoencoder')
    rcae.fit(x_train, x_train, epochs=150, batch_size=128, shuffle=True,
             validation_data=(x_test, x_test), callbacks=[rcae_logger])
    model_to_json(rcae, 'regularized_convolutional_ae.json')
    model_to_json(rcenc, 'regularized_convolutional_enc.json')
    model_to_json(rcdec, 'regularized_convolutional_dec.json')
