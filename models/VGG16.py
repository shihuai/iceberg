from keras.models import Sequential
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import h5py

def get_callbacks(file_path, patience=2):
    # es = EarlyStopping('val_loss', patience=patience, mode='min')
    msave = ModelCheckpoint(file_path, save_best_only=True, monitor='val_loss')
    return [msave];

def get_model(input_shape1=[75, 75, 3], input_shape2=[1], lr=1e-2,
              trainable=True, weights=None):
    bn_model = 0
    # trainable = trainable
    # kernel_regularizer = regularizers.l2(1e-5)
    kernel_regularizer = None
    activation = 'relu'
    # activation = LeakyReLU()

    img_input = Input(shape=input_shape1)
    # angle_input = Input(shape=input_shape2)

    # Block 1
    x = Conv2D(64, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same',
               trainable=trainable, kernel_regularizer=kernel_regularizer,
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # not include the fc layer.
    branch_1 = GlobalMaxPooling2D()(x)
    branch_1 = BatchNormalization(momentum=bn_model)(branch_1)

    branch_2 = GlobalAveragePooling2D()(x)
    # branch_2 = Dropout(0.6)(branch_2)
    branch_2 = BatchNormalization(momentum=bn_model)(branch_2)

    # branch_3 = BatchNormalization(momentum=bn_model)(angle_input)

    # x = (Concatenate()([branch_1, branch_2, branch_3]))
    # x = Dense(1024, activation=activation, kernel_regularizer=kernel_regularizer)(branch_2)
    # x = Dropout(0.6)(x)
    # x = Dense(1024, activation=activation, kernel_regularizer=kernel_regularizer)(x)
    # x = Dropout(0.6)(x)

    # include the fc layer.
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation=activation, name='fc1')(x)
    # x = Dense(4096, activation=activation, name='fc2')(x)

    # branch_2 = BatchNormalization(momentum=bn_model)(angle_input)
    # x = (Concatenate()([x, branch_2]))

    x = (Concatenate()([branch_1, branch_2]))
    x = Dense(1024, activation=activation, kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(0.6)(x)
    # output
    output = Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer)(x)

    # model = Model([img_input, angle_input], output)
    model = Model(img_input, output)
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    # optimizer = SGD(lr=1e-3, momentum=0.9, decay=5*1e-5, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if weights is not None:
        model.load_weights(weights, by_name=True)
        # layer_weights = h5py.File(weights, 'r')
        # for idx in range(len(model.layers)):
        #     model.set_weights()
    print 'have prepared the model.'

    return model