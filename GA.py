import numpy as np
import os, sys
import random
import json
import time

class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)

        self.test_iter = 3

        # 變數
        self.mating_rate = 1.0
        self.num_parents_mating = int(len(self.input) * self.mating_rate)
        self.mutation_rate = 0.6
        self.pop_size = 1000
        self.len_of_gene = len(self.input)
        self.num_generation = 100

        self.chromosomes = None
        self.offspring = None

    def cost(self, ans):
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime

    def compute_fitness(self, chroms):
        #fitness = np.zeros(len(chroms))
        fitness = [0]*len(chroms)
        correct_fitness = [0]*len(chroms)
        best_result = sys.maxsize
        for i in range(len(chroms)):
            for gene in range(self.len_of_gene):
                fitness[i] += self.input[gene][chroms[i][gene]]
            if fitness[i] < best_result :
                best_index = i
                best_result = fitness[i]
        max_obj_fitness = max(fitness)
        for i, obj in enumerate(fitness):
            correct_fitness[i] = max_obj_fitness - fitness[i] + pow(10, -5)
        rank = list(range(len(fitness)))
        rank.sort(key=lambda x:correct_fitness[x]) 
        out_rank = [0]*len(rank)
        for i, x in enumerate(rank):
            out_rank[x] = i
        return correct_fitness, out_rank, best_index

    def roulette_wheel_selection(self, fitness):
        indexs = np.arange(len(fitness))
        total_fitness = np.sum(fitness, axis=0)
        chrom_probs = [fit/total_fitness for fit in fitness]
        
        select = np.random.choice(indexs, self.num_parents_mating, p=chrom_probs)
        
        return select
    def partial_mapped_crossover(self, parent_ids, offspring_size):
        parents = np.uint8([self.chromosomes[id] for id in parent_ids])
        mapped1 = {}
        mapped2 = {}
        
        self.offspring = np.empty(offspring_size)
        for i in range(offspring_size[0]):
            for j in range(offspring_size[1]):
                self.offspring[i][j] = -1
        now = 0
        for k in range(offspring_size[0]):
            rand1 = random.randint(0, self.len_of_gene-1)
            rand2 = random.randint(rand1, self.len_of_gene)
            parentID_1 = now % len(parent_ids)
            parentID_2 = (now+1) % len(parent_ids)
            self.offspring[now][rand1:rand2] = parents[parentID_1][rand1:rand2].copy()
            for m in range(rand1, rand2):
                mapped1[parents[parentID_1][m]] = parents[parentID_2][m]
            for pos in range(len(self.offspring[now])):
                if self.offspring[now][pos] == -1:
                    if parents[parentID_1][pos] in self.offspring[now]:
                        key = parents[parentID_1][pos]
                        while 1:
                            if mapped1[key] in self.offspring[now]:
                                key = mapped1[key].copy()
                            else:
                                self.offspring[now][pos] = mapped1[key].copy()
                                break
                    else:
                        self.offspring[now][pos] = parents[parentID_1][pos].copy()
            now += 1
        
        self.offspring = np.uint(self.offspring)
        return self.offspring

    def order_crossover(self,parent_ids, offspring_size):
        parents = np.uint8([self.chromosomes[id] for id in parent_ids])#[self.chromosomes[id].tolist() for id in parent_ids]
        self.offspring = np.empty(offspring_size)
        
        #crossover_point = np.uint8(offspring_size[1]/2)
        #self.offspring = (([[-1]*offspring_size[1]]*offspring_size[0]).copy()).copy()
        for i in range(offspring_size[0]):
            for j in range(offspring_size[1]):
                self.offspring[i][j] = -1

        now = 0
        for k in range(0,offspring_size[0]*2,2):
            
            rand1 = random.randint(0, self.len_of_gene-2)
            rand2 = random.randint(rand1+1, self.len_of_gene-1)
            parentID_1 = k % len(parent_ids)
            parentID_2 = (k+1) % len(parent_ids)
            self.offspring[now][rand1:rand2] = parents[parentID_1][rand1:rand2].copy()
            parent2 = parents[parentID_2][:].copy()
            parent_left = []
            for i in range(len(parent2)):
                if parent2[i] not in self.offspring[now]:
                    parent_left.append(parent2[i])    
            cnt = 0        
            for pos in range(len(self.offspring[now])):
                if self.offspring[now][pos]!=-1:
                    pass
                else:
                    self.offspring[now][pos] = parent_left[cnt].copy()
                    cnt += 1
            now+=1
        self.offspring = np.uint(self.offspring)
        return self.offspring

    def inversion_mutation(self, offspring_crossover, offspring_size):
        
        rand1 = random.randint(0, self.len_of_gene-2)
        rand2 = random.randint(rand1+1, self.len_of_gene-1)
        offspring_mutation = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            for i in range(self.len_of_gene):
                if i < rand1 or i > rand2:
                    offspring_mutation[k][i] = offspring_crossover[k][i].copy()
                else:
                    offspring_mutation[k][i] = offspring_crossover[k][rand2-(i-rand1)].copy()
        offspring_mutation = np.uint8(offspring_mutation)
        self.offspring = offspring_mutation
        
        return self.offspring

    def scramble_mutation(self, offspring_crossover, offspring_size):
        
        offspring_mutation = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            rand1 = random.randint(0, self.len_of_gene-2)
            rand2 = random.randint(rand1+1, self.len_of_gene-1)
            
            np.random.shuffle(np.uint(offspring_crossover[k][rand1:rand2]))
            offspring_mutation[k][rand1:rand2] = np.uint(offspring_crossover[k][rand1:rand2])
            for i in range(self.len_of_gene):
                if i < rand1 or i >= rand2:
                    offspring_mutation[k][i] = offspring_crossover[k][i].copy()
                
        offspring_mutation = np.uint8(offspring_mutation)
        self.offspring = offspring_mutation.copy()
        return self.offspring


    def new_population(self):
        fitness1, rank1, _ = self.compute_fitness(self.offspring)
        fitness2, rank2, _ = self.compute_fitness(self.chromosomes)
        new_population = np.empty((len(self.chromosomes), len(self.chromosomes[0])))
        for i in range(int(len(self.chromosomes)/2)):
            new_population[i] = self.chromosomes[rank2[rank2.index(i)]]

        length2 = int(len(self.chromosomes)) - int(len(self.chromosomes)/2)
        for i in range(length2):
            new_population[i+int(len(self.chromosomes)/2)] = self.offspring[rank1[rank1.index(i)]]
        self.chromosomes = np.uint8(new_population)
        return new_population

def GA_tester(input):
    solver = Problem(input)
    # initialize
    solver.chromosomes = np.zeros((solver.pop_size, solver.len_of_gene), dtype = int)
    for i in range(solver.pop_size):
        for j in range(solver.len_of_gene):  
            solver.chromosomes[i][j] = j
        np.random.shuffle(solver.chromosomes[i])
    
    best_fitness = -100
    best_cost = sys.maxsize
    best_chromosome = None

    fitness, _, best_index = solver.compute_fitness(solver.chromosomes)
    solver.test_iter = 100
    avg_cost = 0
    avg_time = 0
    for test in range(solver.test_iter):
        start = time.time()
        for generation in range(solver.num_generation):
            # Elitism Selection * roulette_wheel_selection
            parent_ids = solver.roulette_wheel_selection(fitness) # 30 * 0.6
            parents = np.uint8([solver.chromosomes[id] for id in parent_ids])
            
            # crossover & mutation
            solver.offspring = None
            #solver.offspring = solver.order_crossover(parent_ids , offspring_size = [int(solver.pop_size), solver.len_of_gene])
            solver.offspring = solver.partial_mapped_crossover(parent_ids, offspring_size = [int(solver.pop_size), solver.len_of_gene])
            solver.offspring = solver.inversion_mutation(solver.offspring, offspring_size = [int(solver.pop_size), solver.len_of_gene])
            #solver.offspring = solver.scramble_mutation(solver.offspring, offspring_size = [int(solver.pop_size), solver.len_of_gene])
            
            # next generation's population
            solver.new_population()

            # fitness

            fitness, _ , best_index = solver.compute_fitness(solver.chromosomes)
            if fitness[best_index] > float(best_fitness):
                best_fitness = fitness[best_index]
                best_chromosome = solver.chromosomes[best_index]
            if solver.cost(solver.chromosomes[best_index]) < best_cost:
                best_cost = solver.cost(solver.chromosomes[best_index])
                best_chromosome = solver.chromosomes[best_index]
        end = time.time()
        gen_time = end - start
        avg_time += gen_time
        avg_cost += best_cost
    print("After all generation, the best cost is ",best_cost)
    print("Assignment", best_chromosome)
    print("average cost:", avg_cost/solver.test_iter)
    print("average time:", avg_time/solver.test_iter)
    # print('Assignment:', solver.chromosomes[best_index]) 
    # print('Cost:', solver.cost(solver.chromosomes[best_index]))

def GA(input):
    solver = Problem(input)
    # initialize
    solver.chromosomes = np.zeros((solver.pop_size, solver.len_of_gene), dtype = int)
    for i in range(solver.pop_size):
        for j in range(solver.len_of_gene):  
            solver.chromosomes[i][j] = j
        np.random.shuffle(solver.chromosomes[i])
    
    best_fitness = -100
    best_cost = sys.maxsize
    best_chromosome = None

    fitness, _, best_index = solver.compute_fitness(solver.chromosomes)
    start = time.time()
    for generation in range(solver.num_generation):
        # Elitism Selection * roulette_wheel_selection
        parent_ids = solver.roulette_wheel_selection(fitness) 
        parents = np.uint8([solver.chromosomes[id] for id in parent_ids])
        
        # crossover & mutation
        solver.offspring = None
        #solver.offspring = solver.order_crossover(parent_ids , offspring_size = [int(solver.pop_size), solver.len_of_gene])
        solver.offspring = solver.partial_mapped_crossover(parent_ids, offspring_size = [int(solver.pop_size), solver.len_of_gene])
        solver.offspring = solver.inversion_mutation(solver.offspring, offspring_size = [int(solver.pop_size), solver.len_of_gene])
        #solver.offspring = solver.scramble_mutation(solver.offspring, offspring_size = [int(solver.pop_size), solver.len_of_gene])
        
        # next generation's population
        solver.new_population()

        # fitness

        fitness, _ , best_index = solver.compute_fitness(solver.chromosomes)
        if fitness[best_index] > float(best_fitness):
            best_fitness = fitness[best_index]
            best_chromosome = solver.chromosomes[best_index]
        if solver.cost(solver.chromosomes[best_index]) < best_cost:
            best_cost = solver.cost(solver.chromosomes[best_index])
            best_chromosome = solver.chromosomes[best_index]
    end = time.time()
        
    print("Assignment: ", best_chromosome)
    print("Cost: ",best_cost)
    

    # print('Assignment:', solver.chromosomes[best_index]) 
    # print('Cost:', solver.cost(solver.chromosomes[best_index]))


if __name__ == '__main__':

    with open('input.json', 'r') as inputFile:
        data = json.load(inputFile)
        for key in data:
            input = data[key]
   
            GA(input)
    
    