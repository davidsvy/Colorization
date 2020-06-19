import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, Activation 
from tensorflow.keras.layers import UpSampling2D, Lambda, BatchNormalization, Input, Conv2DTranspose
from tensorflow.keras import backend as K
from utils import weighted_sparse_crossentropy
from config import IMG_SHAPE



def get_model(input_shape):
    
    weights = np.load('data/weights_final.npy')
    weights_tf = tf.convert_to_tensor(weights, dtype = tf.float32)
    Q = weights.shape[0]
    
    
    num_filters = 32
    dropout_rate = 0.25
    
    input_ = Input(shape = input_shape)
    
    #====================================================
    #                        Conv1
    #====================================================
    
    x = Conv2D(num_filters,
              kernel_size = 3,
              padding = 'same',
              activation = 'relu',
              name = 'conv1_1') (input_)
    
    x = Conv2D(num_filters,
              kernel_size = 3,
              strides = 2,
              padding = 'same',
              activation = 'relu',
              name = 'conv1_2') (x)
    
    x = BatchNormalization(name = 'conv1_2_norm') (x)
    x = Dropout(dropout_rate, name = 'conv1_2_drop') (x)
    
    #====================================================
    #                        Conv2
    #====================================================
    
    x = Conv2D(2 * num_filters,
              kernel_size = 3,
              padding = 'same',
              activation = 'relu',
              name = 'conv2_1') (x)
    
    x = Conv2D(2 * num_filters,
              kernel_size = 3,
              padding = 'same',
              strides = 2,
              activation = 'relu',
              name = 'conv2_2') (x)
    
    x = BatchNormalization(name = 'conv2_2_norm') (x)
    x = Dropout(dropout_rate, name = 'conv2_2_drop') (x)
    
    #====================================================
    #                        Conv3
    #====================================================
    
    x = Conv2D(4 * num_filters,
              kernel_size = 3,
              padding = 'same',
              activation = 'relu',
              name = 'conv3_1') (x)
    
    x = Conv2D(4 * num_filters,
              kernel_size = 3,
              padding = 'same',
              strides = 1,
              activation = 'relu',
              name = 'conv3_2') (x)
    
    x = BatchNormalization(name = 'conv3_2_norm') (x)
    x = Dropout(dropout_rate, name = 'conv3_2_drop') (x)
    
    
    #====================================================
    #                        Conv4
    #====================================================
    
    x = Conv2D(4 * num_filters,
              kernel_size = 3,
              padding = 'same',
              dilation_rate = 2,
              activation = 'relu',
              name = 'conv4_1') (x)
    
    x = Conv2D(4 * num_filters,
              kernel_size = 3,
              padding = 'same',
              dilation_rate = 2,
              activation = 'relu',
              name = 'conv4_2') (x)
    
    x = BatchNormalization(name = 'conv4_2_norm') (x)
    x = Dropout(dropout_rate, name = 'conv4_2_drop') (x)
    
    
    #====================================================
    #                        Conv5
    #====================================================
    
    x = Conv2DTranspose(4 * num_filters,
              kernel_size = 4,
              padding = 'same',
              strides = 2,
              activation = 'relu',
              name = 'conv5_1') (x)
    
    x = Conv2D(4 * num_filters,
              kernel_size = 3,
              padding = 'same',
              activation = 'relu',
              name = 'conv5_2') (x)
    
    x = Conv2D(4 * num_filters,
              kernel_size = 3,
              padding = 'same',
              activation = 'relu',
              name = 'conv5_3') (x)
    
    x = BatchNormalization(name = 'conv5_3_norm') (x)
    
    #====================================================
    #                        Conv6
    #====================================================
    
    x = Conv2DTranspose(Q,
              kernel_size = 4,
              padding = 'same',
              strides = 2,
              activation = 'relu',
              name = 'conv6_1') (x)
    
    x = Conv2D(Q,
              kernel_size = 3,
              padding = 'same',
              activation = 'relu',
              name = 'conv6_2') (x)
    
    x = Conv2D(Q,
              kernel_size = 2,
              padding = 'same',
              activation = 'relu',
              name = 'conv6_3') (x)
    
    x = Conv2D(Q,
              kernel_size = 1,
              padding = 'same',
              name = 'logits') (x)
    
    
    #====================================================
    #                        Softmax
    #====================================================
    
    output_ = Activation(keras.activations.softmax, name = 'softmax') (x)

    
    model = keras.Model(inputs = [input_], outputs = [output_])
    model.compile(loss = weighted_sparse_crossentropy(weights_tf), optimizer = 'adam')
    
    return model
    
extended_img_shape = IMG_SHAPE + (1, )

model = get_model(extended_img_shape)

if __name__ == '__main__':
    
    model.summary()
    keras.utils.plot_model(model, to_file='data/model_plot.png', show_shapes=True, show_layer_names=True)

