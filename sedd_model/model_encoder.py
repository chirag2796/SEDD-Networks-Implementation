import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import os
import config
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np
import csv
import cv2

def train():
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(10, activation='relu'),
                                    tf.keras.layers.Dense(1024, activation=None)])

    # print(model.summary())
    # exit()
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.save("files\\models\\encoder.h5")

def encode():
    model = tf.keras.models.load_model("files\\models\\encoder.h5")
    encoding_dataset = []
    for i in range(0, 10000):
        image_name = "cat.{}.jpg".format(i)
        img_array = cv2.imread('files\\data\\images\\{}'.format(image_name))
        # img_array = cv2.imread(r'D:\Dev\Datasets\Images\cats_and_dogs\train\train\cat.356.jpg')
        img_array = cv2.resize(img_array, (150, 150))
        img_array = img_array / 255.0
        encoding = model.predict(img_array)
        encoding_dataset.append((image_name, encoding))
        print(i)

    with open("files\\data\\encodings\\encodings.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for encoding in encoding_dataset:
            row = [encoding[0]]
            row += encoding[1].tolist()[0]
            csv_writer.writerow(row)



# train()
encode()