import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

X_train = pd.read_csv("files\\data\\encodings\\encodings.csv", header=None)
image_names = list(X_train[X_train.columns[0]])
X_train.drop(X_train.columns[0], axis=1, inplace=True)
targets = []

img_array = cv2.imread(r'D:\Dev\Datasets\Images\cats_and_dogs\train\train\cat.3002.jpg')
img_array = cv2.resize(img_array, (150, 150))
plt.imshow(img_array)
plt.show()
# cv2.imshow('image', img_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()