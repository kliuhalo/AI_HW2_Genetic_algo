import numpy as np
import os

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






if __name__ == '__main__':
    input = [
        [10, 20, 23,  4],
        [15, 13,  6, 25],
        [ 2, 22, 35, 34],
        [12,  3, 14, 17]
    ]
    crossover_rate = 0.2
    mutation_rate = 0.1
    pop_size = len(input)
    len_of_gene = len(input[0])

    crossover_size = int(pop_size * crossover_rate)
    #####
    if crossover_size % 2 == 1:
        crossover_size -= 1
    #####
    mutation_size = int(pop_size * mutation_rate)

    total_size = pop_size + crossover_size + mutation_size
    

    selected_chromosomes = np.zeros((pop_size, len_of_gene))
    chromosomes = np.zeros((total_size, len_of_gene), dtype = int)
    for i in range(pop_size):
        for j in range(len_of_gene):  
            chromosomes[i][j] = j
        np.random.shuffle(chromosomes[i])
    indexs = np.arange(total_size)
    for i in range(pop_size,total_size):
        for j in range(self.number_of_genes):
            chromosomes[i][j] = -1
    fitness = np.zeros(total_size)
    best_chromosome = np.zeros(len_of_gene)
    best_fitness = 0
    
    # initialize








    # yourAssignment = [3, 2, 0, 1] 

    # solver = Problem(input)
    # print('Assignment:', yourAssignment) 
    # print('Cost:', solver.cost(yourAssignment))
