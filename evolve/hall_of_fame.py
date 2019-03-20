class HallOfFame():
    """
    Data structure for holding the N 'best ever' solutions from an evolutionary run
    """
    def __init__(self, size):
        self.size = size
        self.solutions = []
        self.scores = []

    def getSolution(self, index):
        """
        return the solution in the hall at the specified index
        """
        return self.solutions[index]

    def getFitness(self, index):
        """
        return the fitness score of the solution in the hall at the specified index
        """
        return self.scores[index]

    def updateHall(self, other_solutions):
        """
        replace solutions in the hall with (better?) other solutions
        (or do nothing if none of them belong here)
        """
        merged_solutions = self.solutions + other_solutions[:self.size]
        sorted_solutions = [genome for genome in sorted(merged_solutions, key=lambda x: genome.fitness, reverse=True)]
        self.solutions = sorted_solutions[:self.size]