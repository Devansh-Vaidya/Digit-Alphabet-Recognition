import numpy as np
from tensorflow import keras
from keras import layers


class Data:
    x_train_digit = None
    y_train_digit = None
    x_test_digit = None
    y_test_digit = None

    # The number of classes to predict

    def __init__(self):
        # Load the data and split it into train and test sets
        (x_train_digit, y_train_digit), (x_test_digit, y_test_digit) = keras.datasets.mnist.load_data()
        x_train_digit = x_train_digit.astype("float32") / 255
        y_test_digit = y_test_digit.astype("float32") / 255
        num_classes = 10

        # Add a channels dimension - needed when passing through neural network
        x_train_digit = np.expand_dims(x_train_digit, -1)
        x_test_digit = np.expand_dims(x_test_digit, -1)

        # Convert class vectors to binary class matrices
        y_train_digit = keras.utils.to_categorical(y_train_digit, num_classes=num_classes)
        y_test_digit = keras.utils.to_categorical(y_test_digit, num_classes)

        print(f"x_train shape: {x_train_digit.shape} - y_train shape: {y_train_digit.shape}")


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
                layers.Dense(10, activation="softmax"),
            ]
        )

        self.model.summary()

    def train(self, data, batch_size=128, epochs=15):
        """
        Train the neural network
        :param data: Data that has been prepreocessed and ready to be used for training
        """

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(data.x_train_digit, data.y_train_digit, batch_size=batch_size, epochs=epochs,
                       validation_split=0.1)
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])


def main():
    data = Data()
    neural_network = NeuralNetwork()
    neural_network.train(data, 128, 15)


if __name__ == "__main__":
    main()
