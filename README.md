# Digit and Alphabet Recognition
This project aims to recognize handwritten digits and alphabets using a graphical user interface (GUI) built with Tkinter. The application utilizes the Keras library to obtain the EMNIST dataset and create a machine learning model for recognition.

## Table of Contents
- Prerequisites
- Installation
- Usage
- Dataset
- Model Training
- GUI
- License

## Prerequisites
To use this project, you need to have the Conda installed.

Refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for instructions on installing Conda.

## Installation
To run this project locally, follow these steps:
1. Clone the repository:
```console
git clone https://github.com/Devansh-Vaidya/digit-alphabet-recognition.git
```

2. Navigate to the project directory:
```console
cd digit-alphabet-recognition
```

3. Install the required dependencies mentioned in `requirements.txt`.
```console
conda install --file requirements.txt
```

4. Install `customtkinter` and `extra-keras-datasets` packages using PyPI in Conda.
```console
pip install extra-keras-datasets
pip install customtkinter
```

## Usage
To start the application, run the following command:

```console
python main.py
```

## Dataset
The EMNIST dataset is used for training and testing the recognition model. It consists of handwritten digits and alphabets.

The dataset can be obtained using the `keras` library. The `emnist` module from `keras.datasets` provides convenient functions to load and preprocess the dataset.

## Model Training
The machine learning model for digit and alphabet recognition is built using the Keras library. The model is trained on the EMNIST dataset and optimized using gradient descent.

The `ml.py` file contains the code for model training. It loads the EMNIST dataset, preprocesses the data, defines the model architecture, trains the model, and saves the trained model weights to a file.

To train the model, run the following command:

```console
python ml.py
```

## GUI
The graphical user interface (GUI) is implemented using the Tkinter library. It provides a user-friendly interface for drawing digits or alphabets and obtaining predictions from the trained model.

The `main.py` file contains the code for the GUI. It loads the trained model for predictions, sets up the GUI window, and handles user interactions.

## License
This project is licensed under the MIT License. Feel free to modify and distribute it as per the terms of the license.
