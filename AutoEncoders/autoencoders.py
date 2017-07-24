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
    autoencoder.compile(optimizer='adadelta',
                        loss='binary_crossentropy',
                        metrics=['acc', 'mse'])
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
    autoencoder.compile(optimizer='adadelta',
                        loss='binary_crossentropy',
                        metrics=['acc', 'mse'])
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


def plot_encoded(img, lab=None, n=None, **kwargs):
    if n is None:
        n = np.minimum(img.shape[0], 128)
    max_img_width = kwargs.get('max_img_width', 2)
    max_img_height = kwargs.get('max_img_height', 2)
    max_img_cols = kwargs.get('max_img_cols', 4)
    max_figure_width = kwargs.get('max_figure_width', 16)
    max_figure_height = kwargs.get('max_figure_height', 500)
    sorted_labels_bool = kwargs.get('sorted_labels', False)
    include_cmap = kwargs.get('include_cmap', True)
    cbar_adjust = kwargs.get('colorbar_adjust', .8)
    cbar_left = kwargs.get('colorbar_left', .85)
    cbar_bottom = kwargs.get('colorbar_bottom', .15)
    cbar_width = kwargs.get('colorbar_width', .05)
    cbar_height = kwargs.get('colorbar_height', .7)
    cbar_rect = kwargs.get('colorbar_rect', [cbar_left,
                                             cbar_bottom,
                                             cbar_width,
                                             cbar_height])
    return_fig = kwargs.get('return_fig', False)

    num_img_rows = np.ceil(n / max_img_cols).astype(int)
    num_img_cols = np.minimum(n, max_img_cols).astype(int)
    figure_height = np.minimum(max_img_height * num_img_rows,
                               max_figure_height).astype(int)
    figure_width = np.minimum(num_img_cols * max_img_width,
                              max_figure_width).astype(int)

    img = img[:n, ...]
    img = img.reshape((n, -1, img.shape[-1]))
    if lab is not None:
        lab = lab[:n]
    if sorted_labels_bool:
        print('sorting inputs...')
        label_sort_order = np.argsort(lab)
        lab = lab[label_sort_order]
        img = img[label_sort_order, ...]

    vmin = img.min()
    vmax = img.max()
    fig, axes = plt.subplots(nrows=num_img_rows, ncols=num_img_cols,
                             figsize=(figure_width, figure_height))
    j = 0
    while j < n:
        ax = axes.flat[j]
        cax = ax.imshow(img[j], vmin=vmin, vmax=vmax)
        if lab is not None:
            ax.set_title('label: {}'.format(lab[j]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        j += 1
    while j < num_img_rows * num_img_cols:
        fig.delaxes(axes.flatten()[j])
        j += 1
    if include_cmap:
        fig.subplots_adjust(right=cbar_adjust)
        cbar_ax = fig.add_axes(cbar_rect)
        fig.colorbar(cax, cax=cbar_ax)
    if return_fig:
        return fig
    else:
        return


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    input_shape = x_train.shape[1:]
    scae, scenc, scdec = simple_convolutional_autoencoder(input_shape)
    scae_logname = 'simple_convolutional_ae_' + strftime() + '.log'
    scae_logger = CSVLogger(scae_logname)
    print("Training simple convolutional autoencoder...")
    scae.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True,
             validation_data=(x_test, x_test), callbacks=[scae_logger])
    model_to_json(scae, 'simple_convolutional_ae.json')
    model_to_json(scenc, 'simple_convolutional_enc.json')
    model_to_json(scdec, 'simple_convolutional_dec.json')

    rcae, rcenc, rcdec = regularized_convolutional_autoencoder(input_shape)
    rcae_logname = 'regularized_convolutional_ae_' + strftime() + '.log'
    rcae_logger = CSVLogger(rcae_logname)
    print('Training regularized convolutional autoencoder...')
    rcae.fit(x_train, x_train, epochs=150, batch_size=128, shuffle=True,
             validation_data=(x_test, x_test), callbacks=[rcae_logger])
    model_to_json(rcae, 'regularized_convolutional_ae.json')
    model_to_json(rcenc, 'regularized_convolutional_enc.json')
    model_to_json(rcdec, 'regularized_convolutional_dec.json')
