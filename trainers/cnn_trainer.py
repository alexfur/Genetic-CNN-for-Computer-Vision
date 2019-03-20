class Trainer():
    def __init__(self, model, config, data):
        self.model = model
        self.config = config
        self.data = data
        self.compile_model()

    # compile model
    def compile_model(self):
        print("[INFO] compiling model...")
        self.model.compile(loss="categorical_crossentropy", optimizer='adam',
                      metrics=["accuracy"])

    # train the network
    def train(self):
        print("[INFO] training model...")
        return self.model.fit(self.data['trainX'], self.data['trainY'],
                      validation_data=(self.data['testX'], self.data['testY']),
                      batch_size=self.config['batchSize'], epochs=self.config['numEpochs'])



