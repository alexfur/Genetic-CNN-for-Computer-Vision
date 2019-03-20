
class Genome():
    """
    a Genome is a list of layer dictionaries, each containing CNN hyperparams for the corresponding layer (so a Phenome would be a CNN)
    """
    def __init__(self, conv_layers, dense_layers, genome_mom=None, genome_dad=None, fitness=None):
        self.conv_layers = conv_layers               # [{activation:, dropout:, kernel_size:, filters: }]
        self.dense_layers = dense_layers             # [{activation:}]
        self.layers = conv_layers + dense_layers
        self.parents = [genome_mom, genome_dad]
        self.fitness = 0

    def __repr__(self):
        return str([self.conv_layers + self.dense_layers])