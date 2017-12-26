from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

def get_callbacks(file_path, patience=2):
    # the schedule of stop training the model
    es = EarlyStopping('val_loss', patience=patience, mode='min')
    # the schedule of save the model
    msave = ModelCheckpoint(file_path, save_best_only=True, monitor='val_loss')
    # the schedule of reduce the learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min')

    return [reduce_lr, msave]