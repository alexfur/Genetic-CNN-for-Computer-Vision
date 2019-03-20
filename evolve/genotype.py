
# a Genome is a list of layer dictionaries, each containing CNN hyperparams for the corresponding layer (and a Phenome is a CNN)
class Genome():
    def __init__(self, conv_layers, dense_layers, genome_mom=None, genome_dad=None, fitness=None):
        self.conv_layers = conv_layers               # [{activation:, dropout:, kernel_size:, filters: }]
        self.dense_layers = dense_layers             # [{activation:}]
        self.chromosome = {'conv_layers':conv_layers, 'dense_layers':dense_layers}
        self.layers = conv_layers + dense_layers
        self.parents = [genome_mom, genome_dad]
        self.fitness = 0

    def __repr__(self):
        return str([self.conv_layers + self.dense_layers])


# TODO: create this class like
# https://github.com/harvitronix/neural-network-genetic-algorithm/blob/master/network.py
# or
# https://github.com/jliphard/DeepEvolve/blob/73b2204efc661c5f9abede6d2960b8ed7cdfd46d/genome.py