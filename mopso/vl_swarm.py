import math as m
import numpy as np
import pandas as pd
import copy
from mopso.d2mopso_utils import update_archive, aggregation_value, update_z
from mopso.vl_particle import VLParticle


class VLSwarm(object):
    def __init__(self, chunk_size, type_division, size_division, number_of_particles, problem, localsearch):
        self.TOTAL_LS_CALL = 0
        self.COUNT_LS_FOUND_PBEST = 0
        self.REFRESH_GAP = 10
        self.REASSIGN = False
        self.localsearch = localsearch
        self.w = 0.45
        self.c = 1.49
        self.particles = []
        self.problem = problem
        self._size_division = size_division  # number of divisions
        self._type_division = type_division  # type of different lengths of particles
        self._number_of_particles = number_of_particles
        self._max_size = self.getProblem().get_dimension()
        self._best_size = self._max_size
        self._chunk_size = chunk_size
        self._max_velocity = 0.6
        self._min_velocity = -0.6
        self._Pc = np.zeros(self._number_of_particles)
        self._div_numOfParticles = int(self._number_of_particles / self._size_division)
        self.sizes = np.zeros(self._size_division)
        self._max_layers = self._max_size / self._chunk_size
        self.z = [0.5, 0.5]



        print("**********Initialize all particles************************************")
        print("max length:", self._max_size)

        for j in range(100):
            self.sizes = np.random.random_integers(1, self._type_division, self._size_division)
            flag = True
            for i in range(1, self._type_division + 1):
                eFlag = i in self.sizes
                flag = flag and eFlag
            if flag == True:
                print(j)
                self.sizes = self.sizes * self._chunk_size
                print(self.sizes)
                break
        size = []
        for i in range(int(self._size_division)):
            current_particle_size = int(self.sizes[i])
            size.append(current_particle_size)
            for j in range(self._div_numOfParticles):
                p = VLParticle(current_particle_size)
                f = self.getProblem().fitness(p.position)
                p.setFitness(f)
                p.setPersonalFitness(f)
                self.particles.append(p)
        self._number_of_particles = len(self.particles)

        print("divsion amount:", self._size_division, " particles amount:", self._number_of_particles)
        print("size:", size)
        print("***********Initialization completed***********************************")
        bestidx = np.random.randint(0, self._number_of_particles)
        bestidx = 0
        self._Pc = np.zeros(self._number_of_particles)
        self.refresh_gap_count = np.zeros(self._number_of_particles)
        self.assign_lam()
        self.calculatePc(0)
        self.archive =[]
        self.archive_size = number_of_particles

        self._gbest = VLParticle(self._max_size)

    def renewExemplars(self):
        print("renew Exemplars")
        for i in range(self._number_of_particles):
            self.renewExemplar(i)
            # print(self.getParticle(i).exemplar)

    def renewExemplar(self, p):
        par = self.getParticle(p)
        for d in range(par.size):
            rnd = np.random.uniform(0, 1)
            exempl = p
            if rnd >= self._Pc[p]:
                exempl = p
            else:
                p1 = int(np.round(np.random.uniform(0, 1) * self._number_of_particles))
                attemp = 0
                satify = False
                while satify == False and attemp < self._number_of_particles:
                    p1 = int(np.round(np.random.uniform(0, 1) * (self._number_of_particles - 1)))
                    attemp = attemp + 1
                    satify = ((p1 != p)) and (self.getParticle(p1).size > d)
                if satify == False:
                    p1 = p
                p2 = int(np.round(np.random.uniform(0, 1) * self._number_of_particles))
                attemp = 0
                satify = False
                while satify == False and attemp < self._number_of_particles:
                    p2 = int(np.round(np.random.uniform(0, 1) * (self._number_of_particles - 1)))
                    attemp = attemp + 1
                    satify = ((p2 != p)) and (self.getParticle(p2).size > d)
                if satify == False:
                    p2 = p
                if (aggregation_value(self.getParticle(p1).fitness, self.z,self.getParticle(p1).lam) > aggregation_value(self.getParticle(p2).fitness, self.z,self.getParticle(p2).lam)):
                    exempl = p1
                else:
                    exempl = p2
            self.getParticle(p).setExemplar(d, exempl)
        return True

    def reinit(self):  # find the size that has the best avg pbest fitness
        print("reinit")
        sizeOfDivision = int(self._size_division)
        size = np.zeros(sizeOfDivision)
        avg_fit = np.zeros(sizeOfDivision)
        divSize = int(self._number_of_particles / sizeOfDivision)
        for i in range(int(self._size_division)):
            start_particle_idx = i * divSize
            size[i] = self.getParticle(start_particle_idx).size
            for j in range(self._div_numOfParticles):
                index = start_particle_idx + j
                p = self.getParticle(index)
                avg_fit[i] = avg_fit[i] + aggregation_value(p.personal_fitness,self.z,p.lam)
            avg_fit[i] = avg_fit[i] / divSize
        best_idx = 0
        best_avg_fit = 0.0
        # best avg fit to update max length
        for i in range(int(self._size_division)):
            if best_avg_fit <= avg_fit[i]:
                best_avg_fit = avg_fit[i]
                best_idx = i
        self._best_size = self._gbest.size
        if self._max_size != self._best_size:
            self._max_size = self._max_size - self._chunk_size
            self.ResizeParticles(size)
        else:
            print("Best size unchanged")

    def ResizeParticles(self, size):
        print("Resize all particles according to the best size randomly ", self._max_size)
        sizeOfDivision = int(self._size_division)
        divSize = int(self._div_numOfParticles)
        max_chunk_num = self._max_size / self._chunk_size
        best_chunk_num = self._best_size / self._chunk_size
        chunk_nums = np.zeros(self._size_division)
        new_sizes = np.array(size)
        unit_size = self._max_size / self._type_division
        for i in range(len(size)):
            current_chunk_num = size[i] / self._chunk_size
            if (current_chunk_num != best_chunk_num):
                new_size_f = current_chunk_num * unit_size
                unitnum = int(np.round(new_size_f / self._chunk_size))
                if (unitnum == 0):
                    unitnum = 1
                chunk_nums[i] = unitnum
                new_sizes[i] = chunk_nums[i] * self._chunk_size
        print(new_sizes)
        for i in range(int(self._size_division)):
            new_size = int(new_sizes[i])
            for j in range(self._div_numOfParticles):
                index = i * divSize + j
                cur_size = self.getParticle(index).size
                if cur_size < new_size:
                    for l in range(cur_size, new_size):
                        self.getParticle(index).add_pos(index)
                elif cur_size > new_size:
                    for l in range(new_size, cur_size):
                        self.getParticle(index).remove_pos()
        return True

    def calculatePc(self, type):
        print("calculate Pc")
        if type == 0:
            for i in range(self._number_of_particles):
                # original
                self._Pc[i] = 0.05 + 0.45 * m.exp(10 * (i - 1) / (self._number_of_particles - 1)) / (m.exp(10) - 1)
        elif type == 1:
            for i in range(self._number_of_particles):
                # adaptive learning
                self._Pc[i] = 0.05 + 0.45 * m.exp(10 * (self.rank(i) - 1) / (self._number_of_particles - 1)) / (
                        m.exp(10) - 1)
        return True

    def get_max_size(self):
        return self._max_size

    def averageSize(self):
        sumSize = 0
        for par in self.particles:
            sumSize = sumSize + len(par.position)
        averageSize = sumSize / self._number_of_particles
        return averageSize

    def averageFitness(self):
        averageFitness=sumFitness = [0.0,0.0]
        for par in self.particles:
            sumFitness[0] = sumFitness[0] + par.fitness[0]
            sumFitness[1] = sumFitness[1] + par.fitness[1]
        averageFitness[0] = sumFitness[0]/self._number_of_particles
        averageFitness[1] = sumFitness[1]/self._number_of_particles
        return averageFitness

    def updateVelocityPosition(self, w):
        print("updating all particles.....")
        for par in self.particles:
            positions = par.position
            velocity = par.velocity
            dimensions = len(positions)
            for d in range(dimensions):
                e = int(par.exemplar[d])
                eValue = self.getParticle(e).personal_position[d]
                pValue = positions[d]
                vValue = velocity[d]
                rnd = np.random.uniform(0, 1)
                v = w * vValue + self.c * rnd * (eValue - pValue)
                if v >= self._max_velocity:
                    v = self._max_velocity
                if v <= self._min_velocity:
                    v = self._min_velocity
                velocity[d] = v
                p = positions[d] + velocity[d]
                if p >= 1.0:
                    p = 0.99999999
                if p <= 0.0:
                    p = 0.00000001
                positions[d] = p
            positions.astype('float32')
            velocity.astype('float32')
            par.setPositions(positions)
            par.setVelocitys(velocity)
            f = self.getProblem().fitness(par.position)
            par.setFitness(f)
        return True

    def getParticle(self, index):
        return self.particles[index]

    def rank(self, index):
        swarmFitness = []
        for p in (self.particles):
            print(p.lam)
            a=aggregation_value(p.fitness,self.z,p.lam )
            swarmFitness.append(a)
        return int(pd.Series(swarmFitness).rank(method='max')[index])

    def setProblem(self):
        return True

    def getProblem(self):
        return self.problem

    def setC(self, c):
        self.c = c

    def setW(self, w):
        self.w = w

    def updateFitnessAndLSPbest(self, LOCAL_SEARCH, LS_MAX_TIMES):
        print("updateFitnessAndLSPbest")
        have_new_gbest = False
        if self.REASSIGN:
            self.renewExemplars()
        for p in range(self._number_of_particles):
            # print("update pbest...")
            is_better1 = aggregation_value(self.particles[p].fitness, self.z, self.particles[p].lam) - aggregation_value(self.particles[p].personal_fitness, self.z, self.particles[p].lam)
            if (is_better1 >= 0):
                if (is_better1 > 0):
                    new_position = np.array(self.particles[p].position)
                    new_fitness = self.getProblem().fitness(new_position)
                    self.particles[p].setPersonalPositions(new_position)
                    self.particles[p].setPersonalFitness(new_fitness)
                elif (is_better1 == 0) and (LOCAL_SEARCH):
                    self.TOTAL_LS_CALL = self.TOTAL_LS_CALL + 1
                    LS_FOUND = False
                    for i in range(LS_MAX_TIMES):
                        current_position = np.array(self.particles[p].position)
                        current_fitness = self.particles[p].fitness
                        new_position = np.array(self.localsearch.mutate(current_position))
                        new_fitness = self.getProblem().fitness(new_position)
                        is_better0 = aggregation_value(current_fitness, self.z, self.particles[p].lam) - aggregation_value(new_fitness,
                                                                                                        self.z, self.particles[p].lam)
                        if is_better0 > 0.0:
                            self.COUNT_LS_FOUND_PBEST = self.COUNT_LS_FOUND_PBEST + 1
                            LS_FOUND = True
                            self.particles[p].setPositions(new_position)
                            self.particles[p].setFitness(new_fitness)
                            self.particles[p].setPersonalPositions(new_position)
                            self.particles[p].setPersonalFitness(new_fitness)
            # update reference point z
            update_z(self.z, self.particles[p].fitness)
            # print("update gbest...")
            # update archive
            if (len(self.archive)) < self.archive_size:
                self.archive.append(copy.deepcopy(self.particles[p]))
                have_new_gbest = True
                self.refresh_gap_count[p] = 0
            else:
                if (update_archive(self.particles[p], self.archive)):
                    have_new_gbest = True
                    self.refresh_gap_count[p] = 0
                else:
                    self.refresh_gap_count[p] = self.refresh_gap_count[p] + 1
                    if (self.refresh_gap_count[p] == self.REFRESH_GAP):
                        self.REASSIGN = True
                        self.refresh_gap_count[p] = 0
            self._gbest = copy.deepcopy(self.archive[0])

        return have_new_gbest

    def assign_lam(self):

        lamSet = []
        for i in range(10):
            lam = np.random.uniform(0.1, 1, 2)
            lamSet.append(lam)
        lamMul = 0
        index = 0
        print(lamSet)
        for p in range(self._number_of_particles):
            index = 0
            for i in range(len(lamSet)):
                new_lamMul = aggregation_value(self.particles[p].fitness, self.z, lamSet[i])
                if (new_lamMul > lamMul):
                    lamMul = new_lamMul
                    index = i
            lam = np.array(lamSet[index])
            self.particles[p].setLambda(lam)
            del lamSet[index]

        return False