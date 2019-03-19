import yaml
from data_loaders.data_loader import load_and_preprocess_data
from models.cnn_model import CNNModel
from trainers.cnn_trainer import Trainer
from utils.model_utils import *

#model = load_model('cnn_model.h5')

config = yaml.safe_load(open('configs/config.yaml'))    # get config

data = load_and_preprocess_data(config)                 # get fashion mnist data

trainer = Trainer(CNNModel.build(), config, data)       # initialise trainer with a newly initialised model

generate_model_summary(trainer.model)                   # summary of the model structure

training_history = trainer.train()                      # train the model, save historical stats over epochs

generate_classification_report(trainer.model, data)     # make predictions on test set and generate corresponding classification report

generate_training_stats_plot(config, training_history)  # plot the training loss and accuracy

generate_demo(data, trainer.model)                      # save demo montage classification pic

save_model(trainer.model, "cnn_model")          # save the model to disk