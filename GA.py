import numpy as np
import os, sys
import random
class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)
        self.GA_algo = GA_algorithm()

    def cost(self, ans):
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime
    
#def initialize():


input = [
    [10, 20, 23,  4],
    [15, 13,  6, 25],
    [ 2, 22, 53, 34],
    [12,  3, 14, 17]
]

# START
# Generate the initial population
# Compute fitness
# REPEAT
#     Selection
#     Crossover
#     Mutation
#     Compute fitness
# UNTIL population has converged
# STOP

def compute_fitness(fitness, chromosomes):
    fitness = np.zeros(len(chromosomes))
    for i in range(len(chromosomes)):
        for gene in range(len_of_gene):
            fitness[i] += input[gene][chromosomes[i][gene]]
    print(fitness)

    return fitness

def update_best_solution(fitness):

    pass

def roulette_wheel_selection(indexs, fitness):
    
    total_fitness = np.sum(fitness, axis=0)
    print("total fitness", total_fitness)
    chrom_probs = [1 -(fit / total_fitness) for fit in fitness]
    print("chrom probs", chrom_probs)
    total_fitness = np.sum(chrom_probs, axis=0)
    chrom_probs =[(prob/total_fitness) for prob in chrom_probs]
    print(chrom_probs)
    print(len(chrom_probs),len(indexs))
    select = np.random.choice(indexs, num_parents, p=chrom_probs)
    print(select)
    return select

def crossover(parent_ids, offspring_size):
    parents = np.uint8([chromosomes[id] for id in parent_ids])
    print("parents")
    print(parents)

    offspring = np.empty(offspring_size)
    print(np.shape(offspring))
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parentID_1 = k % len(parent_ids)
        parentID_2 = k % len(parent_ids)
        offspring[k][crossover_point:] = parents[parentID_1][0:crossover_point]
        offspring[k][crossover_point:] = parents[parentID_2][crossover_point:]
        offspring = np.uint(offspring)
    print("offspring ", " = ",offspring)

    return offspring

def inversion_mutation(offspring_crossover, offspring_size):
    print("============")
    rand1 = random.randint(0, len_of_gene-2)
    rand2 = random.randint(rand1+1, len_of_gene-1)
    # rand1 = 0
    # rand2 = 3
    offspring_mutation = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        for i in range(len_of_gene):
            if i < rand1 or i > rand2:
                offspring_mutation[k][i] = offspring_crossover[k][i]
            else:
                offspring_mutation[k][i] = offspring_crossover[k][rand2-(i-rand1)]
    offspring_mutation = np.uint8(offspring_mutation)
    print(offspring_mutation)
    return offspring_mutation

# 變數
num_parents = 4
mutation_rate = 0.1
pop_size = 10
len_of_gene = len(input[0])
num_generation = 5




# initialize
selected_chromosomes = np.zeros((pop_size, len_of_gene))
chromosomes = np.zeros((pop_size, len_of_gene), dtype = int)
for i in range(pop_size):
    for j in range(len_of_gene):  
        chromosomes[i][j] = j
    np.random.shuffle(chromosomes[i])

print("chromosomes:")
print(chromosomes)
print()

fitness = np.zeros(pop_size)
best_chromosome = np.zeros(len_of_gene)
best_fitness = 0
indexs = np.arange(pop_size)

fitness = compute_fitness(fitness, chromosomes)

parent_ids = roulette_wheel_selection(indexs, fitness)


parents = np.uint8([chromosomes[id] for id in parent_ids])
print("parents")
print(parents)
offspring = crossover(parent_ids , offspring_size = [int(pop_size*0.2), len_of_gene])
offspring = inversion_mutation(offspring, offspring_size = [int(pop_size*0.2), len_of_gene])
chromosomes = np.asarray(chromosomes + offspring)
print(np.shape(chromosomes))
# for generation in (num_generation):

#     # selection

#     # crossover & mutation

#     # fitness
#     compute_fitness(fitness, chromosomes)
    








    # yourAssignment = [3, 2, 0, 1] 

    # solver = Problem(input)
    # print('Assignment:', yourAssignment) 
    # print('Cost:', solver.cost(yourAssignment))
