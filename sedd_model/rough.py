import pandas as pd
import numpy as np
import cv2

X_train = pd.read_csv("files\\data\\encodings\\encodings.csv", header=None)
image_names = list(X_train[X_train.columns[0]])
X_train.drop(X_train.columns[0], axis=1, inplace=True)
targets = []
for i in range(len(image_names)):
    image_name = image_names[i]
    img_array = cv2.imread('files\\data\\images\\{}'.format(image_name))
    img_array = cv2.resize(img_array, (150, 150))
    img_array = img_array / 255.0
    targets.append([img_array.flatten()])

targets = np.vstack(targets)
print(targets.shape)
print(type(targets))