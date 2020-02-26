import pdb

import numpy as np

from mopso.vl_swarm import VLSwarm


class VLPSO(object):
    def __init__(self, para1, problem, localsearch):
        # default config and settings
        self.config = para1
        self.max_number_of_iterations = self.config['iteration']
        self.type_of_division = self.config['layers']
        self.size_of_division = self.type_of_division
        self.div_numOfParticles = int(self.config['population'] / self.type_of_division)
        self.c = 1.49445
        self.w = 0.9 - np.random.uniform(0, 1) * 0.5
        self.dimensions = self.config['dimension']
        self.chunk_size = self.config['chunk_size']  # weight size of each layer
        self.MAX_ITER_STAG = int(2 * self.max_number_of_iterations / 3)
        self.LOCAL_SEARCH = self.config['LOCAL_SEARCH']
        self.REINIT = self.config['REINIT']
        self.LS_MAX_ITER = self.config['LS_MAX_ITER']
        self.problem = problem
        self.localsearch = localsearch
        self.number_of_particles = self.size_of_division * self.div_numOfParticles
        self.generation_output = []
        self.best_output = []

        print("Pop size:", self.number_of_particles, "\nMax iter:", self.max_number_of_iterations)

    def VELPSO(self, seed=1):
        np.random.seed(seed)
        swarm = VLSwarm(self.chunk_size, self.type_of_division, self.size_of_division, self.number_of_particles,
                        self.problem, self.localsearch)
        swarm.setC(self.c)
        swarm.setW(self.w)
        swarm.COUNT_LS_FOUND_PBEST = 0
        print("**************************************************************")
        iter = 0
        nbr_iter_not_improve = 0
        self.LS_MAX_ITER = 100
        local_search_flag = self.LOCAL_SEARCH
        found_new_gbest = swarm.updateFitnessAndLSPbest(False, self.LS_MAX_ITER)
        if found_new_gbest:
            nbr_iter_not_improve = 0
        else:
            nbr_iter_not_improve = nbr_iter_not_improve + 1
        swarm.calculatePc(0)
        swarm.renewExemplars()
        best_candidate = None
        best_fitness = swarm.getProblem().get_worst_fitness()

        while iter < self.max_number_of_iterations:
            gbest_size = len(swarm.archive[0].getPersonalPositions())
            avg_fitn = swarm.averageFitness()
            avg_size = swarm.averageSize()
            print("iter %d, gbest fitness: [%.3f ,%.3f](%d)|Avg fit: [%.3f,%.3f], Avg size : %.3f\n" % (
                iter, swarm.archive[0].getPersonalFitness()[0],swarm.archive[0].getPersonalFitness()[1], gbest_size, avg_fitn[0],avg_fitn[1], avg_size))
            self.best_output.append(swarm.archive[0].getPersonalFitness())
            self.generation_output.append(avg_fitn)
            if self.REINIT and nbr_iter_not_improve >= self.MAX_ITER_STAG:
                '''print("number of no improvement:", nbr_iter_not_improve)'''
                swarm.reinit()
                swarm.updateFitnessAndLSPbest(self.LOCAL_SEARCH, self.LS_MAX_ITER)
                swarm.calculatePc(1)
                swarm.renewExemplars()
                nbr_iter_not_improve = 0
            w = 0.9 - ((iter / self.max_number_of_iterations) * 0.5)
            swarm.updateVelocityPosition(w)
            iter = iter + 1
            local_search_flag = self.LOCAL_SEARCH and ((iter < 20) and (iter % 2 == 0))
            found_new_gbest = swarm.updateFitnessAndLSPbest(local_search_flag, iter)
            if found_new_gbest:
                nbr_iter_not_improve = 0
                print("archive updated!")
            else:
                nbr_iter_not_improve = nbr_iter_not_improve + 1

        best_candidate = np.array(swarm.archive[0].getPersonalPositions())
        best_fitness = swarm.archive[0].getPersonalFitness()
        LS_UPDATE = 0
        for i in range(self.LS_MAX_ITER):
            new_candidate = np.array(self.localsearch.mutate(best_candidate))
            if self.problem.fitness(new_candidate) > best_fitness:
                best_candidate = new_candidate
                LS_UPDATE = LS_UPDATE + 1
        print(swarm.archive[0])
        print(best_candidate)
        print(best_fitness)
        return best_candidate, best_fitness, self.generation_output, self.best_output
