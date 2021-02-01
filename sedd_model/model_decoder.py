import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import config
import cv2
import tensorflow as tf

class Decoder:
    class Plotter:
        @staticmethod
        def decode_and_plot(generator, encoding):
            dim = (1, 1)
            figsize = (8, 8)
            generated_images = generator.predict(encoding)
            generated_images = generated_images.reshape(config.NUMBER_OF_IMAGES, config.IMAGE_WIDTH,
                                                        config.IMAGE_HEIGHT, config.COLOR_MULTIPLIER)
            # generated_images = generated_images * 255
            generated_images = np.round(generated_images)
            generated_images = generated_images.astype(int)
            print(generated_images)
            plt.figure(figsize=figsize)
            for i in range(generated_images.shape[0]):
                plt.subplot(dim[0], dim[1], i + 1)
                plt.imshow(generated_images[i], interpolation='nearest')
                plt.axis('off')
            plt.tight_layout()
            plt.show()

        @staticmethod
        def plot_generated_images(generator):
            dim = (1, 1)
            figsize = (8, 8)
            noise = np.random.normal(loc=0, scale=1, size=[config.NUMBER_OF_IMAGES, config.NUMBER_OF_IMAGES])
            generated_images = generator.predict(noise)
            generated_images = generated_images.reshape(config.NUMBER_OF_IMAGES, config.IMAGE_WIDTH,
                                                        config.IMAGE_HEIGHT)
            # print(generated_images)
            plt.figure(figsize=figsize)
            for i in range(generated_images.shape[0]):
                plt.subplot(dim[0], dim[1], i + 1)
                plt.imshow(generated_images[i], interpolation='nearest')
                plt.axis('off')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def adam_optimizer():
        return Adam(lr=0.0002, beta_1=0.5)

    @staticmethod
    def test_image(image_path, encoder_model, decoder_model):
        img_array = cv2.imread(image_path)
        img_array = cv2.resize(img_array, (150, 150))
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)
        # img_array = img_array.astype(np.float)
        encoding = encoder_model.predict(img_array)
        # exit()
        Decoder.Plotter.decode_and_plot(decoder_model, encoding)

    @staticmethod
    def create_generator():
        generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1024, input_dim=config.ENCODING_SIZE),
            tf.keras.layers.LeakyReLU(0.2),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(units=512, input_dim=config.ENCODING_SIZE),
            tf.keras.layers.LeakyReLU(0.2),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(units=512, input_dim=config.ENCODING_SIZE),
            tf.keras.layers.LeakyReLU(0.2),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(units=512, input_dim=config.ENCODING_SIZE),
            tf.keras.layers.LeakyReLU(0.2),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(units=config.IMAGE_HEIGHT * config.IMAGE_WIDTH * config.COLOR_MULTIPLIER, activation='relu')
        ])

        generator.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        # generator.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mse'])
        return generator

    @staticmethod
    def train(generator_model):
        X_train = pd.read_csv("files\\data\\encodings\\encodings.csv", header=None)
        image_names = list(X_train[X_train.columns[0]])
        X_train.drop(X_train.columns[0], axis=1, inplace=True)
        targets = []
        for i in range(len(image_names)):
            image_name = image_names[i]
            img_array = cv2.imread('files\\data\\images\\{}'.format(image_name))
            img_array = cv2.resize(img_array, (150, 150))
            # img_array = img_array / 255.0
            targets.append([img_array.flatten()])
        targets = np.vstack(targets)
        print(targets.shape)
        # print(X_train.shape)
        # exit()


        history = generator_model.fit(X_train, targets, epochs=4, batch_size=32, verbose=1, validation_split=0.1)
        generator_model.save("files\\models\\decoder.h5")
        Decoder.plot_training_history(history)

    @staticmethod
    def plot_training_history(history):
        # list all data in history
        print(history.history.keys())

        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(111)

        ax1.plot(builds, y_stack[0, :], label='Component 1', color='c', marker='o')
        ax1.plot(builds, y_stack[1, :], label='Component 2', color='g', marker='o')
        ax1.plot(builds, y_stack[2, :], label='Component 3', color='r', marker='o')
        ax1.plot(builds, y_stack[3, :], label='Component 4', color='b', marker='o')

        plt.xticks(builds)
        plt.xlabel('Builds')

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15, 1))
        ax1.grid('on')

        print(history.history['mean_squared_error'])
        print(history.history['val_mean_squared_error'])


        plt.grid('on')
        # summarize history for accuracy
        plt.plot(history.history['mean_squared_error'], label='mean_squared_error', marker='o')
        plt.plot(history.history['val_mean_squared_error'], label='val_mean_squared_error', marker='o')
        plt.title('Decoder mean_squared_error')
        plt.ylabel('mean_squared_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()



generator = Decoder.create_generator()
Decoder.train(generator)
# exit()

#  Use After Traning
# encoder_model = tf.keras.models.load_model("files\\models\\encoder.h5")
# decoder_model = tf.keras.models.load_model("files\\models\\decoder.h5")
#
# print(decoder_model.summary())
# exit()
# image_path = r"D:\Dev\Datasets\Images\cats_and_dogs\train\train\cat.3002.jpg"
# Decoder.test_image(image_path, encoder_model, decoder_model)
# Decoder.plot_generated_images(generator)