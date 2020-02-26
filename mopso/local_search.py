import numpy as np


class LocalSearch(object):
    def __init__(self, parameters, problem):
        self.parameters = parameters
        self.problem = problem

    def mutate(self, candidate):
        mutateFlag = False
        can = np.array(candidate)
        mutatePercentage = self.parameters['percentage']
        for i in range(len(can)):
            if mutatePercentage > np.random.uniform(0, 1):
                can[i] = 1 - can[i]
                mutateFlag = True
        return can
