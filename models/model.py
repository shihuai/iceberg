from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def get_callbacks(file_path, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode='min')
    msave = ModelCheckpoint(file_path, save_best_only=True)
    return [es, msave];

def get_model(input_shape1=[75, 75, 3], input_shape2=[1], weights=None):
    bn_model = 0
    # bn_model = 0.99
    p_activation = 'elu'
    input_1 = Input(shape=input_shape1, name='X_1')
    # input_2 = Input(shape=input_shape2, name='angle')

    img_1 = Conv2D(16, kernel_size=(3,3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size=(3,3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPool2D()(img_1)

    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPool2D()(img_2)

    # img_concat = (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))
    img_concat = (Concatenate()([img_1, img_2]))
    dense_layer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(256, activation=p_activation)(img_concat)))
    dense_layer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(64, activation=p_activation)(dense_layer)))
    output = Dense(1, activation='sigmoid')(dense_layer)

    # model = Model([input_1, input_2], output)
    model = Model(input_1, output)
    optimizer = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model