from keras.preprocessing.image import ImageDataGenerator

def get_image_generator(horizontal_flip=True,
                        vertical_flip=True,
                        width_shift_range=0.0,
                        height_shift_range=0.0,
                        zca_whitening=True,
                        zoom_range=0.2,
                        rotation_range=20):

    return ImageDataGenerator(horizontal_flip=horizontal_flip,
                            vertical_flip=vertical_flip,
                            width_shift_range=width_shift_range,
                            height_shift_range=height_shift_range,
                            zca_whitening=zca_whitening,
                            zoom_range=zoom_range,
                            rotation_range=rotation_range)