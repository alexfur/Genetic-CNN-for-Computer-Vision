# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class model:
    @staticmethod
    def build():
        # initialize the model
        model = Sequential()

        model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='linear', input_shape=(28,28,1))) # Input layer (28x28)
        model.add(MaxPooling2D(pool_size=2))                                                                    # Convolution2D
        model.add(Dropout(0.0))                                                                                 # Dropout

        model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.05))

        model.add(Conv2D(filters=32, kernel_size=7, padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.05))

        model.add(Flatten())
        model.add(Dense(64, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dropout(0.10))

        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model
