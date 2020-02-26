import numpy as np


class VLParticle(object, ):
    def __init__(self, particle_size):
        self.feasible = False
        self.fitness = [0.0, 0.0]
        self.size = particle_size
        self.maxVelocity = 0.6
        self.minVelocity = -0.6
        self.exemplar = np.zeros(self.size)
        self.lam=np.random.uniform(0.1, 1, 2)
        self.position = np.random.uniform(0, 1, self.size)
        self.velocity = np.random.uniform(self.minVelocity, self.maxVelocity, self.size)
        self.personal_position = np.random.uniform(0, 1, self.size)
        self.personal_fitness = 0.0

    def setPosition(self, index, p):
        if (len(self.position)) > index:
            self.position[index] = p
            return True
        else:
            return False

    def setPositions(self, p):
        self.position = p

    def setPersonalPosition(self, index, p):
        if (len(self.personal_position)) > index:
            self.personal_position[index] = p
            return True
        else:
            return False

    def setPersonalPositions(self, p):
        self.personal_position = p

    def setVelocity(self, index, v):
        if (len(self.velocity)) > index:
            self.velocity[index] = v
            return True
        else:
            return False

    def setVelocitys(self, v):
        self.velocity = v

    def setPersonalFitness(self, f):
        self.personal_fitness = f
        return True

    def setExemplar(self, index, e):
        if len(self.exemplar) > index:
            self.exemplar[index] = e
            return True
        else:
            return False

    def setFitness(self, f):
        self.fitness = f
        return True
    def setLambda(self,l):
        self.lam=np.array(l)

    def getLambda(self):
        return self.lam
    def getFitness(self):
        return self.fitness

    def getPersonalFitness(self):
        return self.personal_fitness

    def getPosition(self, index):
        if (len(self.position)) > index:
            return self.position[index]
        else:
            return False

    def getPersonalPosition(self, index):
        if (len(self.personal_position)) > index:
            return self.personal_position[index]
        else:
            return -1

    def getPersonalPositions(self):
        return self.personal_position

    def getVelocity(self, index):
        if (len(self.velocity) > index):
            return self.velocity[index]
        else:
            return False

    def getVelocitys(self):
        return self.velocity

    def getExemplars(self):
        return self.exemplar

    def getSize(self):
        return self.size

    def remove_pos(self):
        self.position = self.position[:-1]
        self.velocity = self.velocity[:-1]
        self.exemplar = self.exemplar[:-1]
        self.personal_position = self.personal_position[:-1]
        self.size = self.size - 1
        return True

    def add_pos(self,index):
        p = np.random.uniform(0, 1)
        v = np.random.uniform(self.minVelocity, self.minVelocity)
        e = index
        np.append(self.position, p)
        np.append(self.velocity, v)
        np.append(self.exemplar, e)
        np.append(self.personal_position, p)
        self.size = self.size + 1
        return True

    def copyParticle(self, p):
        self.position = np.array(p.getPositions())
        self.velocity = np.array(p.getVelocitys())
        self.exemplar = np.array(p.getExemplars())
        self.lam=np.array(p.getLambda())
        #    self.size = int(p.getSize())
        #    self.fitness =float(p.getFitness())
        self.personal_position = np.array(p.getPersonalPositions())
        #    self.personal_fitness = float(p.getPersonalFitness())

    def getPositions(self):
        return self.position

    def getPersonalPositions(self):
        return self.personal_position
