import pandas as pd
import numpy as np
import cv2

df_train = pd.read_json('../data/train.json')
def get_scaled_imgs(df):
    imgs = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = ((band_1 + band_2) / 2);

        a = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.min()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

X_train = get_scaled_imgs(df_train)
Y_train = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na', 0)
X_angle_train = np.array(df_train.inc_angle)
# idx_tr = np.where(df_train.inc_angle > 0)
# Y_train = Y_train[idx_tr[0]]
# X_train = X_train[idx_tr[0], ...]

print X_train.shape
for img in X_train:
    # img = img * 255
    # img = img.astype(np.uint8)
    print img.dtype
    print img.shape
    print img
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('img', img)
    cv2.waitKey(0)