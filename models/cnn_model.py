# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from evolve.genotype import Genome

class CNNModel:
    @staticmethod
    def buildForEvolution(genome):
        # initialize the model
        model = Sequential()

        conv_layers = genome.conv_layers
        dense_layers = genome.dense_layers


        # Layer 1
        model.add(Conv2D(filters=conv_layers[0]['filters'], kernel_size=conv_layers[0]['kernel_size'], padding='same', activation=conv_layers[0]['activation'], input_shape=(28,28,1)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(conv_layers[0]['dropout']))

        # Layer 2
        model.add(Conv2D(filters=conv_layers[1]['filters'], kernel_size=conv_layers[1]['kernel_size'], padding='same', activation=conv_layers[1]['activation']))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(conv_layers[1]['dropout']))

        # Layer 3
        model.add(Conv2D(filters=conv_layers[2]['filters'], kernel_size=conv_layers[2]['kernel_size'], padding='same', activation=conv_layers[2]['activation']))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(conv_layers[2]['dropout']))

        # Layer 4
        model.add(Flatten())
        model.add(Dense(64, activation=dense_layers[0]['activation']))

        # Layer 5
        model.add(Dense(10, activation=dense_layers[1]['activation']))

        model.add(Activation('softmax'))


        # return the constructed network architecture
        return model

    @staticmethod
    def buildNoEvolution():
        # initialize the model
        model = Sequential()

        # Layer 1
        model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='linear',
                         input_shape=(28, 28, 1)))  # Input layer (28x28)
        model.add(MaxPooling2D(pool_size=2))  # Convolution2D
        model.add(Dropout(0.0))  # Dropout

        # Layer 2
        model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.05))

        # Layer 3
        model.add(Conv2D(filters=32, kernel_size=7, padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.05))

        # Layer 4
        model.add(Flatten())
        model.add(Dense(64, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dropout(0.10))

        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model
