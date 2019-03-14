"""This module defines a densely-connected neural network model to
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
from keras.models import Sequential              # class of neural networks
from keras.layers.core import Dense, Activation  # type of layers
from keras.utils import np_utils                 # extra tools
# from keras.models import load_model


class FashionMNISTModel:

    def __init__(self):
        """Instantiate convolutional neural net model.

        Instead of initiating a new model, one could load an existing model
        and continue training e.g self.model = load_model('my_model_NN.h5')
        and bypass the model compilation in the rest of this __init__ method.
        """

        self.num_classes = 4

        # First, declare a model with a sequential architecture
        model = Sequential()

        # first layer of 500 nodes and 784 inputs (the pixels of the image)
        # 'relu' activation function to use on the nodes of that first layer
        model.add(Dense(500, input_shape=(784,)))
        model.add(Activation('relu'))

        # Second hidden layer with 300 nodes
        model.add(Dense(300))
        model.add(Activation('relu'))

        # Output layer with 4 categories (using softmax activation)
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=["accuracy"])
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
