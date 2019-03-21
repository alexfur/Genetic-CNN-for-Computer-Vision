import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def save_model(model, results_dir):        # cnn_model.h5
    model.save(results_dir+'cnn_model.h5')