import numpy as np
import os, sys
import random
import json

class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)

        # 變數
        self.mating_rate = 0.6
        self.num_parents_mating = int(len(self.input) * self.mating_rate)
        self.mutation_rate = 0.6
        self.pop_size = len(self.input)
        self.len_of_gene = len(self.input)
        self.num_generation = 15

        self.chromosomes = None
        self.offspring = None

    def cost(self, ans):
        print("ans", ans)
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime

    def compute_fitness(self, chroms):
        #fitness = np.zeros(len(self.chromosomes))
        #fitness = np.zeros(len(chroms))
        fitness = [0]*len(chroms)
        best_result = sys.maxsize
        #for i in range(len(self.chromosomes)):
        for i in range(len(chroms)):
            for gene in range(self.len_of_gene):
                fitness[i] += self.input[gene][chroms[i][gene]]#self.input[gene][self.chromosomes[i][gene]]
            if fitness[i] < best_result :
                best_index = i
                best_result = fitness[i]
        rank = list(range(len(fitness)))#np.zeros(len(chroms))
        rank.sort(key=lambda x:fitness[x]) 
        out_rank = [0]*len(rank)
        for i, x in enumerate(rank):
            out_rank[x] = i
        # fitness = np.uint8(fitness)
        #print("fitness", fitness)
        return fitness, out_rank, best_index

    def roulette_wheel_selection(self, fitness):
        indexs = np.arange(len(fitness))
        total_fitness = np.sum(fitness, axis=0)
        #print("total fitness", total_fitness)
        chrom_probs = [1 -(fit / total_fitness) for fit in fitness]
        #print("chrom probs", chrom_probs)
        total_fitness = np.sum(chrom_probs, axis=0)
        chrom_probs =[(prob/total_fitness) for prob in chrom_probs]
        #print(chrom_probs)
        #print(len(chrom_probs),len(indexs))
        select = np.random.choice(indexs, self.num_parents_mating, p=chrom_probs)
        #print("select",select)
        return select

    def order_crossover(self,parent_ids, offspring_size):
        parents = np.uint8([self.chromosomes[id] for id in parent_ids])#[self.chromosomes[id].tolist() for id in parent_ids]
        #print("parents: ", parents)
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
        # rand1 = 0
        # rand2 = 3
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

    def new_population(self):
        fitness1, rank1, _ = self.compute_fitness(self.offspring)
        fitness2, rank2, _ = self.compute_fitness(self.chromosomes)
        # print("new population")
        # print(fitness1, fitness2)
        # print(rank1, rank2)
        new_population = np.empty((len(self.chromosomes), len(self.chromosomes[0])))
        for i in range(int(len(self.chromosomes)/2)):
            # print(rank2[rank2.index(i)])
            # print("findfind",self.chromosomes[rank2[rank2.index(i)]])
            new_population[i] = self.chromosomes[rank2[rank2.index(i)]]

        length2 = int(len(self.chromosomes)) - int(len(self.chromosomes)/2)
        for i in range(length2):
            new_population[i+int(len(self.chromosomes)/2)] = self.offspring[rank1[rank1.index(i)]]
        self.chromosomes = np.uint8(new_population)
        #print("new population",self.chromosomes)
        return new_population

def GA(input):
    solver = Problem(input)
    # initialize
    #selected_chromosomes = np.zeros((pop_size, solver.len_of_gene))
    
    solver.chromosomes = np.zeros((solver.pop_size, solver.len_of_gene), dtype = int)
    for i in range(solver.pop_size):
        for j in range(solver.len_of_gene):  
            solver.chromosomes[i][j] = j
        np.random.shuffle(solver.chromosomes[i])
    
    #best_chromosome = np.zeros(solver.len_of_gene)
    best_fitness = 0

    fitness, _, best_index = solver.compute_fitness(solver.chromosomes)
    #print("input", input)
    #print("fitness", fitness)
    for generation in range(solver.num_generation):
        # traditional selection
        parent_ids = solver.roulette_wheel_selection(fitness) # 30 * 0.6
        parents = np.uint8([solver.chromosomes[id] for id in parent_ids])
        
        # Elitism Selection


        # crossover & mutation
        solver.offspring = None
        solver.offspring = solver.order_crossover(parent_ids , offspring_size = [int(solver.pop_size), solver.len_of_gene])
        #print("Order crossover offspring", solver.offspring)
        #print("1",solver.offspring)
        solver.offspring = solver.inversion_mutation(solver.offspring, offspring_size = [int(solver.pop_size), solver.len_of_gene])
        #print("Inversion mutation offspring", solver.offspring)
        #print("2",solver.offspring)
        # next generation's population
        
        solver.new_population()
        # fitness
        fitness, _ , best_index = solver.compute_fitness(solver.chromosomes)
        print("generation", generation)
        print("pop size", solver.pop_size)
        print('Assignment:',solver.chromosomes[best_index])
        print('Cost',fitness[best_index])

    # print('Assignment:', solver.chromosomes[best_index]) 
    # print('Cost:', solver.cost(solver.chromosomes[best_index]))


if __name__ == '__main__':
    with open('input.json', 'r') as inputFile:
        data = json.load(inputFile)
        for key in data:
            input = data[key]
            
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
            #GA(input)
    # input = [
    #         [10, 20, 23,  4],
    #         [15, 13,  6, 25],
    #         [ 2, 22, 53, 34],
    #         [12,  3, 14, 17]
    #         ] 
    input = [[0.43045255, 0.78681387, 0.07514408, 0.72583933, 0.52916145, 0.87483212, 0.34701621],
    [0.68704291, 0.45392742, 0.46862110, 0.67669006, 0.23817468, 0.87520581, 0.67311418],
    [0.38505150, 0.05974168, 0.11388629, 0.28978058, 0.66089373, 0.92592403, 0.70718757],
    [0.24975701, 0.16937649, 0.42003672, 0.88231235, 0.74635725, 0.59854858, 0.88631100],
    [0.64895582, 0.58909596, 0.99772334, 0.85522575, 0.33916707, 0.72873479, 0.26826203],
    [0.47939038, 0.88484586, 0.05122520, 0.83527995, 0.37219939, 0.20375257, 0.50482283],
    [0.58926554, 0.45176739, 0.25217475, 0.83548120, 0.41687026, 0.00293049, 0.23939052]]
    input = [[0.71773280, 0.28980792, 0.86571783, 0.44026587, 0.53155829],
    [0.30556295, 0.56751479, 0.75442822, 0.62446877, 0.30992529],
    [0.83717620, 0.52213939, 0.54137934, 0.15001555, 0.70178034],
    [0.53279199, 0.08006661, 0.70693305, 0.29315974, 0.69018493],
    [0.03981310, 0.25511235, 0.94795653, 0.41611858, 0.50587076]]
    # input =[[10, 20, 23, 4],
    # [15, 13, 6, 25],
    # [ 2, 22, 35, 34],
    # [12, 3, 14, 17]]

    GA(input)