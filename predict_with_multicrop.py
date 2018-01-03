from tools.get_image_generator import get_image_generator
from models import VGG16, model, ResNet50, Inception_v3
from tools.preprocess_img import read_data
import pandas as pd
import numpy as np

def predict_test(file_path, model, datagen=None,
                 batch_size=32, crop_times=1, X_test=None):

    model.load_weights(file_path)
    predictions = []
    count = 0

    for img in X_test:
        img = np.array([img])
        for idx in range(crop_times):
            random_seed = np.random.random_integers(0, 10000)
            if idx == 0:
                prediction = model.predict_generator(datagen.flow(img, None,
                                                                   shuffle=False,
                                                                   batch_size=batch_size,
                                                                   seed=random_seed),
                                                      len(img))
            else:
                prediction += model.predict_generator(datagen.flow(img, None,
                                                                   shuffle=False,
                                                                   batch_size=batch_size,
                                                                   seed=random_seed),
                                                      len(img))
            print prediction.shape
                # print predictions.shape
        # print X_test.shape

        prediction /= crop_times
        predictions.append(prediction[0, 0])
        if count % 1000 == 0:
            print 'have predicted {}/{} files.'.format(count, X_test.shape[0])
        count += 1

    return np.array(predictions)

if __name__=="__main__":
    height = 224
    width = 224
    file_list, X_test, X_angle_test, _ = read_data('data/test.json',
                                                   height=height, width=width)
    print X_test.shape

    datagen = get_image_generator(horizontal_flip=True,
                                  vertical_flip=True,
                                  width_shift_range=0.0,
                                  height_shift_range=0.0,
                                  zca_whitening=True,
                                  zoom_range=0.2,
                                  rotation_range=20)

    file_path = "./model_weights_1.hdf5"
    prediction_1 = predict_test(file_path,
                                ResNet50.get_model(input_shape1=[height, width, 3],
                                                   input_shape2=[1]),
                                datagen=datagen, batch_size=32, crop_times=5,
                                X_test=X_test[:10, :, :, :])
    print prediction_1
    print prediction_1.shape
    # file_path = "./model_weights_2.hdf5"
    # prediction_2 = predict_test(file_path,
    #                             model.get_model(input_shape1=[height, width, 3],
    #                                             input_shape2=[1]),
    #                             X_test)

    # prediction = (prediction_1 + prediction_2) / 2
    prediction = prediction_1
    submission = pd.DataFrame({'id': file_list[:10],
                               'is_iceberg': prediction.reshape((prediction.shape[0]))})
    submission.to_csv("./submission_test.csv", index=False)