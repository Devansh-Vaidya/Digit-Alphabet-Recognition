import numpy as np
from keras import layers
from matplotlib import pyplot as plt
from tensorflow import keras
from extra_keras_datasets import emnist

class Data:
    def __init__(self):
        self.x_train_digit = None
        self.y_train_digit = None
        self.x_test_digit = None
        self.y_test_digit = None

        # Load the data and split it into train and test sets
        (self.x_train_digit, self.y_train_digit), (
            self.x_test_digit, self.y_test_digit) = emnist.load_data(type='bymerge')
        self.x_train_digit = self.x_train_digit.astype("float32") / 255
        self.x_test_digit = self.x_test_digit.astype("float32") / 255
        num_classes = 47

        # Add a channels dimension - needed when passing through neural network
        self.x_train_digit = np.expand_dims(self.x_train_digit, -1)
        self.x_test_digit = np.expand_dims(self.x_test_digit, -1)

        # Convert class vectors to binary class matrices
        self.y_train_digit = keras.utils.to_categorical(self.y_train_digit, num_classes)
        self.y_test_digit = keras.utils.to_categorical(self.y_test_digit, num_classes)

        print(f"x_train shape: {self.x_train_digit.shape} - y_train shape: {self.y_train_digit.shape}")


class NeuralNetwork:
    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(47, activation="softmax"),
            ]
        )

        self.model.summary()

    def train(self, data, batch_size=128, epochs=15):
        self.model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
        training_history = self.model.fit(data.x_train_digit, data.y_train_digit, batch_size=batch_size, epochs=epochs,
                                          validation_split=0.2)
        score = self.model.evaluate(data.x_test_digit, data.y_test_digit, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        return training_history

    @staticmethod
    def plot_learning_curve(training):
        plt.plot(training.history['loss'])
        plt.plot(training.history['val_loss'])
        plt.title('Learning Curve')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def main():
    data = Data()
    neural_network = NeuralNetwork()
    training = neural_network.train(data, 128, 50)
    neural_network.plot_learning_curve(training)

    # Save the trained model
    neural_network.model.save('digit_recognition_model')


if __name__ == "__main__":
    main()
