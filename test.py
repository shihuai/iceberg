from models import VGG16, model, ResNet50
from tools.preprocess_img import read_data
import pandas as pd

# height = 224
# width = 224
# file_list, X_test, X_angle_test, _ = read_data('data/test.json',
#                                             height=height, width=width)
# print X_test.shape
# file_path = './model_weights.hdf5'
# model = get_model(input_shape1=[height, width, 3],
#                   input_shape2=[1])
# model.load_weights(file_path)
# prediction = model.predict([X_test, X_angle_test], verbose=1, batch_size=32)
# submission = pd.DataFrame({'id': file_list,
#                            'is_iceberg': prediction.reshape((prediction.shape[0]))})
# submission.to_csv("./submission.csv", index=False)

def predict_test(file_path, model, X_test):
    model.load_weights(file_path)
    prediction = model.predict(X_test, verbose=1, batch_size=32)

    return prediction

if __name__ == "__main__":
    height = 224
    width = 224
    file_list, X_test, X_angle_test, _ = read_data('data/test.json',
                                                   height=height, width=width)
    print X_test.shape

    file_path = "./model_weights_1.hdf5"
    prediction_1 = predict_test(file_path,
                                ResNet50.get_model(input_shape1=[height, width, 3],
                                                input_shape2=[1]),
                                X_test)
    # file_path = "./model_weights_2.hdf5"
    # prediction_2 = predict_test(file_path,
    #                             model.get_model(input_shape1=[height, width, 3],
    #                                             input_shape2=[1]),
    #                             X_test)

    # prediction = (prediction_1 + prediction_2) / 2
    prediction = prediction_1
    submission = pd.DataFrame({'id': file_list,
                               'is_iceberg': prediction.reshape((prediction.shape[0]))})
    submission.to_csv("./submission.csv", index=False)