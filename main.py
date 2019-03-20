import os, yaml, logging
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from data_loaders.data_loader import load_and_preprocess_data
from models.cnn_model import CNNModel
from trainers.cnn_trainer import Trainer
from utils.model_utils import *
from evolve.GA import Evolution

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.INFO)

config = yaml.safe_load(open('configs/config.yaml'))
numGenerations = config['numGenerations']
data = load_and_preprocess_data(config)                                                                                 # get fashion mnist data

if config['evolvingMode']:
    resultsDir = ''
    if config['intelligentSurvival']:
        resultsDir = 'results/evolution-intelligent-survival/'
    else:
        resultsDir = 'results/evolution-plain/'

    evolution = Evolution(config, data)
    evolution.initialise_population()
    for gen in range(numGenerations):
        evolution.evolvePopulation(gen)

    best_evolved_genome = evolution.hall_of_fame.getSolution(0)

    trainer = Trainer(CNNModel.buildForEvolution(best_evolved_genome), config, data)
    generate_model_summary(trainer.model)
    training_history = trainer.train()
    generate_classification_report(trainer.model, data)
    generate_training_stats_plot(config, training_history, resultsDir)
    generate_demo(data, trainer.model, resultsDir)
    save_model(trainer.model, resultsDir)
    score = trainer.model.evaluate(data['trainX'], data['trainY'], verbose=0)
    print("Score: "+str(score[1]))
else:
    resultsDir = 'results/no-evolution/'
    trainer = Trainer(CNNModel.buildNoEvolution(), config, data)                                                        # initialise trainer with a newly initialised model
    generate_model_summary(trainer.model)                                                                               # summary of the model structure
    training_history = trainer.train()                                                                                  # train the model, save historical stats over epochs
    generate_classification_report(trainer.model, data)                                                                 # make predictions on test set and generate corresponding classification report
    generate_training_stats_plot(config, training_history, resultsDir)                                                  # plot the training loss and accuracy
    generate_demo(data, trainer.model, resultsDir)                                                                      # save demo montage classification pic
    save_model(trainer.model, resultsDir)                                                                               # save the model to disk
    score = trainer.model.evaluate(data['trainX'], data['trainY'], verbose=0)                                           # compute model score (accuracy). This can be compared against best evolved solution's score.
    print("Score: "+str(score[1]))                                                                                      # 1=accuracy, 0=loss




# model = load_model('results/cnn_model.h5')
# plot_model(model, to_file='model.png')