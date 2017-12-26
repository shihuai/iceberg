from keras.preprocessing.image import ImageDataGenerator
from models import VGG16, model, ResNet50
from tools.preprocess_img import read_data, data_augment
from sklearn.model_selection import train_test_split

# height = 75
# width = 75
# train_mode = True
# file_list, X_train, X_angle_train, y_train = read_data('data/train.json',
#                                             height=height, width=width,
#                                             train_mode=True)
# X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train,
#                                                                                     X_angle_train,
#                                                                                     y_train,
#                                                                                     random_state=123,
#                                                                                     train_size=0.75)
# # X_train, X_angle_train, y_train = data_augment(X_train, X_angle_train, y_train)
# print X_train.shape
# print X_valid.shape
#
# file_path = './model_weights.hdf5'
# # pre_trained_model = './model_weights.hdf5'
# pre_trained_model = './pre_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# callbacks = get_callbacks(file_path=file_path, patience=5)
# model = get_model(input_shape1=[height, width, 3],
#                   input_shape2=[1],
#                   weights=pre_trained_model)
#
# datagen = ImageDataGenerator(horizontal_flip=True,
#                                  vertical_flip=True,
#                                  width_shift_range=0.,
#                                  height_shift_range=0.,
#                                  channel_shift_range=0,
#                                  zoom_range=0.2,
#                                  rotation_range=10)
#
# model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
#                     epochs=30, steps_per_epoch=len(X_train) / 32,
#                     validation_data=(X_valid, y_valid),
#                     verbose=1, callbacks=callbacks)
#
# # model.fit([X_train, X_angle_train], y_train,
# #           epochs=25, batch_size=16,
# #           validation_data=([X_valid, X_angle_valid], y_valid),
# #           callbacks=callbacks)

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
    print y_train.sum()
    X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train,
                                                                                        X_angle_train,
                                                                                        y_train,
                                                                                        random_state=123,
                                                                                        train_size=0.9)
    print X_train.shape
    print X_valid.shape
    print X_train[0, :, :, 0]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True,
                                 width_shift_range=0.,
                                 height_shift_range=0.,
                                 channel_shift_range=0,
                                 zoom_range=0.2,
                                 rotation_range=10)

    file_path = "./model_weights_1.hdf5"
    pre_trained_model = './pre_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    callbacks = VGG16.get_callbacks(file_path)
    train_models(file_path,
                 ResNet50.get_model(input_shape1=[height, width, 3],
                                 input_shape2=[1], lr=1e-5,
                                 trainable=True,
                                 weights=pre_trained_model),
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
    # callbacks = VGG16.get_callbacks(file_path)
    # train_models(file_path,
    #              VGG16.get_model(input_shape1=[height, width, 3],
    #                              input_shape2=[1], lr=1e-5,
    #                              trainable=True,
    #                              weights=pre_trained_model),
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