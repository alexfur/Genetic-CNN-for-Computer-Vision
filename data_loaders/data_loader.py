from keras.datasets import fashion_mnist
from keras.utils import np_utils

def load_and_preprocess_data(config=None):

    # grab the Fashion MNIST dataset (if this is your first time running
    # this the dataset will be automatically downloaded)
    print("[INFO] loading Fashion MNIST...")
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()           # clothing item, label

    trainX = trainX[:1000]
    trainY = trainY[:1000]
    testX = testX[:1000]
    testY = testY[:1000]

    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # scale data to the range of [0, 1]
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    # one-hot encode the training and testing labels
    trainY = np_utils.to_categorical(trainY, 10)
    testY = np_utils.to_categorical(testY, 10)

    # initialize the label names
    labelNames = ["top", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    return {'trainX':trainX, 'trainY':trainY, 'testX':testX, 'testY':testY, 'labelNames':labelNames}
