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
    images = []

    for i in np.random.choice(np.arange(0, len(data['testY'])), size=(16,)):
        probs = model.predict(data['testX'][np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = data['labelNames'][prediction[0]]
        image = (data['testX'][i] * 255).astype("uint8")
        color = (0, 255, 0)
        if prediction[0] != np.argmax(data['testY'][i]):
            color = (0, 0, 255)

        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2)

        images.append(image)

    montage = build_montages(images, (96, 96), (4, 4))[0]
    cv2.imwrite(results_dir+"demo_montage.png", montage)

def save_model(model, results_dir):        # cnn_model.h5
    model.save(results_dir+'cnn_model.h5')