import copy

import numpy as np

t=5

def calculate_crowding_distance(archive, chroms_obj_record):
    front=[]
    i=0
    for a in (archive):
        front.append(i)
        i=i+1
    distance = {m: 0 for m in front}
    for o in range(2):
        obj = {m: chroms_obj_record[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)
        distance[sorted_keys[0]] = distance[sorted_keys[len(front) - 1]] = 999999999999
        for i in range(1, len(front) - 1):
            if len(set(obj.values())) == 1:
                distance[sorted_keys[i]] = distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]] = distance[sorted_keys[i]] + (
                        obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) / (
                                                   obj[sorted_keys[len(front) - 1]] - obj[sorted_keys[0]])

    return distance

def selection(population_size, archive, chroms_obj_record, total_chromosome):
    N = 0
    new_pop = []
    while N < population_size:
        distance = calculate_crowding_distance(archive, chroms_obj_record)
        sorted_cdf = sorted(distance, key=distance.get)
        sorted_cdf.reverse()
        for j in sorted_cdf:
            if len(new_pop) == population_size:
                break
            new_pop.append(j)
        break

    population_list = []
    for n in new_pop:
        population_list.append(total_chromosome[n])

    return population_list

def norm_distance(v):
    dimension=len(v)
    distance=0
    for i in range(dimension):
        distance=distance+v[i]*v[i]
    distance=np.sqrt(distance)
    return distance

def T_vector_multiple(v1,v2):
    dimension=len(v1)
    v=[]
    assert(dimension==len(v2))
    for i in range(dimension):
        v.append(v1[i]*v2[i])
    return v
def T_multiple(v,m):
    dimension = len(v)
    vec=[]
    for i in range(dimension):
        vec.append(v[i] * m)
    return vec
def T_minus(v1,v2):
    dimension = len(v1)
    dimension2= len(v2)
    if(dimension!=dimension2):
        return False
    v = []
    for i in range(dimension):
        v.append(v1[i]- v2[i])
    return v

def T_minus_diff(v1,v2):
    dimension1 = len(v1)
    dimension2 = len(v2)
    dimension = 0
    append_index=0
    if (dimension1 < dimension2):
        dimension = dimension1
        append_index=2
    elif (dimension2<dimension1):
        dimension = dimension2
        append_index=1
    v = []
    i=0
    for i in range(dimension):
        v.append(v1[i] - v2[i])
    if(append_index==1):
        for j in range(i,dimension1):
            v.append(v1[j])
    if(append_index==2):
        for j in range(i,dimension2):
            v.append(v2[j])

    return v


def d_one(F,z,l):
    l_norm=norm_distance(l)
    d1_norm=norm_distance(T_vector_multiple(T_minus(F,z),l))
    d1=d1_norm/l_norm
    return d1
def d_two(F,z,l):
    m1=T_minus(F,z)
    m2=T_multiple(l,d_one(F,z,l)/norm_distance(l))
    d2=norm_distance(T_minus(m1,m2))
    return d2
def aggregation_value(F,z,l):
    d1=d_one(F,z,l)
    d2=d_two(F,z,l)*t
    return d1+d2

def update_z(z,p):
    dimension=len(z)
    F=p.fitness
    if(dimension!=len(F)):
        return False
    for i in range(dimension):
        if(F[i]>z[i]):
            z[i]=F[i]
    return z

def d2crowd_vector(p,arhive):
    size=len(arhive)
    cd=[0,0]
    cd1=0#solution space
    cd2=0#objective space
    for i in range(size):
        cd1=cd1+norm_distance(T_minus_diff(p.position,arhive[i].position))
        cd2=cd2+norm_distance(T_minus(p.fitness,arhive[i].fitness))
    cd[0]=cd1
    cd[1]=cd2

    return cd

def update_z(z,new_z):
    if new_z[0]>z[0]:
        z[0]=new_z[0]
    if new_z[1]>z[1]:
        z[1]=new_z[1]

def update_archive(p,archive):
    size=len(archive)
    CD=[]
    rank=[]
    temp_archive= copy.deepcopy(archive)
    can=copy.deepcopy(p)
    temp_archive.append(can)
    for par in (temp_archive):
        CD.append(d2crowd_vector(par,temp_archive))
    archive_list = selection(size,temp_archive, CD, temp_archive)
    if p not in archive_list:
        return False
    else:
        archive=archive_list
        return True
