import yaml
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from data_loaders.data_loader import load_and_preprocess_data
from models.cnn_model import CNNModel
from trainers.cnn_trainer import Trainer
from utils.model_utils import *
from keras.models import load_model
from keras.utils import plot_model
import logging
from evolve.GA import Evolution
from evolve.genotype import Genome

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger().setLevel(logging.INFO)

config = yaml.safe_load(open('configs/config.yaml'))
numGenerations = config['numGenerations']
data = load_and_preprocess_data(config)                  # get fashion mnist data

evolution = Evolution(config, data)
evolution.initialise_population()

for gen in range(numGenerations):
    evolution.evolvePopulation(gen)



# trainer = Trainer(CNNModel.build(), config, data)       # initialise trainer with a newly initialised model
#
# generate_model_summary(trainer.model)                   # summary of the model structure
#
# training_history = trainer.train()                      # train the model, save historical stats over epochs
#
# generate_classification_report(trainer.model, data)     # make predictions on test set and generate corresponding classification report
#
# #generate_training_stats_plot(config, training_history)  # plot the training loss and accuracy
#
# #generate_demo(data, trainer.model)                      # save demo montage classification pic
#
# #save_model(trainer.model, "cnn_model")          # save the model to disk
#
# score = trainer.model.evaluate(data['trainX'], data['trainY'], verbose=0)
# print("ACCURACY: "+str(score[1]))                            # # 1 is accuracy. 0 is loss.



# model = load_model('results/cnn_model.h5')
# plot_model(model, to_file='model.png')