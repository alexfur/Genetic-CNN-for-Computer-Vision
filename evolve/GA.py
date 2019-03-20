from trainers.cnn_trainer import Trainer
from models.cnn_model import CNNModel
import random
from evolve.genotype import Genome

class Evolution():
    def __init__(self, config, data):
        self.genomes = []                                   # population of genomes (each one is a set of CNN hyperparameters)
        self.data = data                                    # TODO: Is data list necessary here?
        self.config = config
        self.numGenerations = config['numGenerations']
        self.populationSize = config['populationSize']
        self.gene_types = list(config['hyperparams'])


    def initialise_population(self):                        # initialise population with random parameter choices
        for i in range(0, self.populationSize):

            # first do random params for convolutional layers (all possible parameters [gene types])
            conv_layers = []
            for layer_num in range(self.config['numConvLayers']):
                conv_layer = {}
                for gene_type in self.config['hyperparams']:                                    # i.e. activation is a gene type
                    gene_val = random.choice(self.config['hyperparams'][gene_type])
                    conv_layer.update({gene_type:gene_val})
                conv_layers.append(conv_layer)

            # now do random params for dense layers (only activation paramater [gene type])
            dense_layers = []
            for layer_num in range(self.config['numDenseLayers']):
                dense_layer = {}
                gene_val = random.choice(self.config['hyperparams']['activation'])
                dense_layer.update({'activation':gene_val})
                dense_layers.append(dense_layer)

            self.genomes.append(Genome(conv_layers, dense_layers))

    def train_and_score(genome, self):
        ...
        #trainer = Trainer(CNNModel.build(activation=), config, data)

    # gene = paramater
    def mutate_one_gene(self, genome):                                                  # TODO: account for random new_gene_val being the same as the old val (while loop)
        layer_to_mutate = random.choice(genome.layers)                                  # which layer to mutate?
        index_layer_to_mutate = genome.layers.index(layer_to_mutate)                    # index of chosen layer to mutate
        gene_type_to_mutate = random.choice(list(layer_to_mutate))                      # mutate which param? Activation? Dropout?
        new_gene_val = random.choice(self.config['hyperparams'][gene_type_to_mutate])   # choose a random new value for the gene
        genome.layers[index_layer_to_mutate][gene_type_to_mutate] = new_gene_val        # set new value of gene we're mutating


    def crossover(self, genomeMom, genomeDad):                                          # TODO: this crossover logic might be too splicey
        """ Create a child genome by mating two parent genomes
            - child's conv layer: split between mother and father
            - child's dense layer: activation function chosen randomly from mother or father """

        parents = [genomeMom, genomeDad]
        random.shuffle(parents)                                                         # shuffle parents so both get a fair chance to give out slightly more
                                                                                        # conv layer genes since there are an odd number of conv layers.
        child_conv_layers = []
        for cl in range(self.config['numConvLayers']):                                  # crossover conv layers
            if (cl+1)%2 == 0:
                child_conv_layers.append(parents[0].conv_layers[cl])
            else:
                child_conv_layers.append(parents[1].conv_layers[cl])

        child_dense_layers = []
        for dl in range(self.config['numDenseLayers']):                                 # crossover dense layers
            if (dl+1)%2 == 0:
                child_dense_layers.append(parents[0].dense_layers[dl])
            else:
                child_dense_layers.append(parents[1].dense_layers[dl])

        return Genome(child_conv_layers, child_dense_layers, genome_mom=genomeMom, genome_dad=genomeDad)                            # return child genome