from tools.get_image_generator import get_image_generator
from models import VGG16, model, ResNet50, Inception_v3, DensNet121, ResNet101, ResNet101_v2
from tools.preprocess_img import read_data, data_augment
from sklearn.model_selection import train_test_split
from models.callback_functions import get_callbacks


def train_models(file_path, model, datagen, epoches, batch_size, train_data, valied_data):
    X_train, y_train = train_data
    X_valied, y_valied = valied_data

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        epochs=epoches, steps_per_epoch=len(X_train) / batch_size,
                        validation_data=(X_valied, y_valied),
                        verbose=1, callbacks=callbacks)


if __name__=="__main__":

    height = 224
    width = 224
    train_mode = True
    file_list, X_train, X_angle_train, y_train = read_data('data/train.json',
                                            height=height, width=width,
                                            train_mode=True)
    # print y_train.sum()
    X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train,
                                                                                        X_angle_train,
                                                                                        y_train,
                                                                                        random_state=123,
                                                                                        train_size=0.9)
    print X_train.shape
    print X_valid.shape
    print X_train[0, :, :, 0]

    datagen = get_image_generator(horizontal_flip=True,
                                 vertical_flip=True,
                                 width_shift_range=0.0,
                                 height_shift_range=0.0,
                                 # zca_whitening=True,
                                 zoom_range=0.2,
                                 rotation_range=20)

    file_path = "./model_weights_1.hdf5"
    pre_trained_model = './pre_models/resnet101_weights_tf.h5'
    callbacks = get_callbacks(file_path)
    train_models(file_path,
                 ResNet101_v2.get_model(input_shape1=[height, width, 3],
                                 input_shape2=[1], lr=1e-4,
                                 trainable=True,
                                 weights=pre_trained_model,
                                 optimizers='adam'),
                 datagen, 50, 16,
                 (X_train, y_train),
                 (X_valid, y_valid))

    # finetune the pre-trained VGG16
    # train_models(file_path,
    #                VGG16.get_model(input_shape1=[height, width, 3],
    #                                input_shape2=[1], lr=1e-5,
    #                                trainable=True,
    #                                weights=pre_trained_model),
    #                datagen, 35, 16,
    #                (X_train, y_train),
    #                (X_valid, y_valid))

    # finetune all the layer of CNN
    # file_path = "./model_weights_2.hdf5"
    # pre_trained_model = './model_weights_1.hdf5'
    # callbacks = get_callbacks(file_path)
    # train_models(file_path,
    #              ResNet50.get_model(input_shape1=[height, width, 3],
    #                              input_shape2=[1], lr=1e-5,
    #                              trainable=True,
    #                              weights=pre_trained_model,
    #                              optimizers='adam'),
    #              datagen, 40, 32,
    #              (X_train, y_train),
    #              (X_valid, y_valid))

    # train different CNN model
    # file_path = "./model_weights_2.hdf5"
    # # pre_trained_model = './pre_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # callbacks = VGG16.get_callbacks(file_path)
    # train_models(file_path,
    #              model.get_model(input_shape1=[height, width, 3],
    #                              input_shape2=[1]),
    #              datagen, 50, 32,
    #              (X_train, y_train),
    #              (X_valid, y_valid))