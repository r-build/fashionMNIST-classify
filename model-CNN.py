"""This module defines a convolutional neural network model to
classify subset of the fashion-MNIST data, and assumes only
four classes of image.

Example use:
    my_model = FashionMNISTModel()
    X_data = np.load('X_data.npy')
    y_data = np.load('y_data.npy')
    X_train, y_train = my_model.preprocess_training_data(X_data, y_data)
    my_model.fit(X_train, y_train)
    X_test = my_model.preprocess_unseen_data(X)
    preds = my_model.predict(X_test)
    print("### Your predictions ###", preds)
"""


import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
# from keras.models import load_model


class FashionMNISTModel:
    """Define convolutional neural net to classify 4 classes of
    fashion-MNIST dataset
    """

    def __init__(self):
        """Instantiate convolutional neural net model.

        Instead of initiating a new model, one could load an existing model
        and continue training e.g self.model = load_model('my_model_CNN.h5')
        and bypass the model compilation in the rest of this __init__ method.
        """

        self.num_classes = 4

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                         input_shape=(28, 28, 1), activation='relu',
                         name='C2D_1'))
        model.add(Conv2D(32, (3, 3), activation='relu', name='C2D_2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))  # regularisation

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                         name='C2D_3'))
        model.add(Conv2D(64, (3, 3), activation='relu', name='C2D_4'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))  # regularisation

        model.add(Flatten())
        model.add(Dense(256, activation='relu', name='D_1'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax', name='D_2'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def preprocess_training_data(self, X, y):
        """Pre-process training data"""

        X = X.reshape(X.shape[0], 28, 28, 1)  # reshape images for keras
        X /= 255.0  # normalise each image's pixels on (0,1)
        return X, y

    def fit(self, X, y):
        """fit model"""

        self.batch_size = int(X.shape[0]/100)
        self.epochs = 30

        # split training data into training and validation sets
        # stratify on y in case y is not distributed randomly amongst rows
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.1, random_state=42)

        # convert integer class (0-3) to 4 categories (each 0-1)
        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)

        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=2,
                       validation_data=(X_test, y_test))

    def preprocess_unseen_data(self, X):
        """Pre-process unseen test data"""

        X = X.reshape(X.shape[0], 28, 28, 1)  # reshape images for keras
        X /= 255.0  # normalise images on (0,1)
        return X

    def predict(self, X):
        """ predict the classes of test images"""

        def get_correct_class(y_pred):
            answers = np.zeros(len(y_pred), dtype=np.int)
            for idx, result in enumerate(y_pred):
                objectclass = np.argmax(result)
                print(objectclass)
                answers[idx] = objectclass
            return answers

        y_probs = self.model.predict(X)
        y_classes = get_correct_class(y_probs)
        return y_classes
