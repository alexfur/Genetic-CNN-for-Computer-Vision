import matplotlib
matplotlib.use("Agg")
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import classification_report

def generate_model_summary(model):
    model.summary()

# TODO: make this save to file called "classification_report.txt"
def generate_classification_report(model, data):
    # make predictions on the test set
    preds = model.predict(data['testX'])

    # show a nicely formatted classification report
    print("[INFO] evaluating network...")
    print(classification_report(data['testY'].argmax(axis=1), preds.argmax(axis=1),
                            target_names=data['labelNames']))


# plot the training loss and accuracy
def generate_training_stats_plot(config, trainer, results_dir):
    N = config['numEpochs']
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), trainer.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), trainer.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), trainer.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), trainer.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(results_dir+"plot.png")


def generate_demo(data, model, results_dir):
    # initialize our list of output images
    images = []

    # randomly select a few testing fashion items
    for i in np.random.choice(np.arange(0, len(data['testY'])), size=(16,)):
        # classify the clothing
        probs = model.predict(data['testX'][np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = data['labelNames'][prediction[0]]

        # extract the image from the testData
        image = (data['testX'][i] * 255).astype("uint8")

        # initialize the text label color as green (correct)
        color = (0, 255, 0)

        # prediction is incorrect if it doesn't match the test data at that index
        if prediction[0] != np.argmax(data['testY'][i]):
            color = (0, 0, 255)

        # merge the channels into one image and resize the image from
        # 28x28 to 96x96 so we can better see it and then draw the
        # predicted label on the image
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2)

        # add the image to our list of output images
        images.append(image)

    # construct the montage for the images
    montage = build_montages(images, (96, 96), (4, 4))[0]

    # show the output montage
    cv2.imwrite(results_dir+"demo_montage.png", montage)


def save_model(model, results_dir):        # cnn_model.h5
    model.save(results_dir+'cnn_model.h5')