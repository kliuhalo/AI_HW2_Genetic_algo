import numpy as np
import os, sys
import random
import json
import time
 
append = 0

def permutation(ele, l ,r):
    global append
    if l==r:
        g = ele.copy()
        ans[append] = g
        append+=1
    else:
        for i in range(l ,r+1):
            ele[l], ele[i] = ele[i], ele[l]
            permutation(ele, l+1, r)

            ele[l], ele[i] = ele[i] , ele[l]
            

def cal_cost(ans):
    
    best_cost = sys.maxsize

    for i, a in enumerate(ans):
        cost = 0
        for i, aa in enumerate(a):
            cost += input[i][aa]
        if cost < best_cost:
            best_cost = cost
            best_chromosome = a
    return best_chromosome, best_cost

def BF(input):
    len_of_gene = len(input)
    total = 1
    for i in range(1, len_of_gene+1):
        total *= i
    global ans
    ans = [0]*total

    indexs = np.arange(len_of_gene)
    permutation(indexs, 0, len_of_gene-1)

    best_chromosome, best_cost = cal_cost(ans)
    return best_chromosome, best_cost
    


if __name__ == '__main__':
    
    # with open('input.json', 'r') as inputFile:
    #     data = json.load(inputFile)
    #     for key in data:
    #         input = data[key]
    input = [
    [10, 20, 23,  4],
    [15, 13,  6, 25],
    [ 2, 22, 53, 34],
    [12,  3, 14, 17]
    ]
    input = [
    [0.26300727684204517, 0.48513471953446996, 0.8491417036699047], 
    [0.7518785807425733, 0.029752222747783996, 0.5887209536993653], 
    [0.7761974553100254, 0.19546118308946114, 0.6158427400193519]
    ]
    input = [[6, 10, 4, 8, 5, 9, 4, 6, 10, 5, 4, 8, 7, 8, 3], [10, 7, 5, 2, 9, 0, 5, 5, 6, 4, 9, 0, 1, 1, 8], [5, 6, 9, 0, 4, 8, 4, 3, 5, 4, 10, 2, 7, 4, 7], [1, 3, 7, 9, 7, 10, 0, 7, 8, 10, 8, 4, 6, 3, 6], [0, 3, 10, 6, 6, 6, 9, 3, 9, 2, 2, 3, 0, 3, 10], [4, 3, 7, 10, 3, 3, 9, 0, 8, 1, 9, 9, 7, 0, 7], [10, 10, 9, 7, 8, 10, 0, 3, 0, 7, 8, 7, 6, 3, 7], [10, 6, 2, 3, 0, 9, 1, 1, 1, 2, 8, 4, 1, 0, 1], [7, 3, 2, 6, 3, 0, 9, 2, 1, 3, 5, 7, 0, 6, 3], [5, 2, 5, 4, 0, 2, 6, 9, 1, 8, 8, 3, 4, 3, 0], [2, 2, 10, 6, 8, 10, 5, 10, 10, 4, 0, 1, 10, 4, 10], [7, 3, 2, 1, 5, 0, 4, 2, 4, 4, 2, 2, 8, 0, 0], [5, 8, 5, 3, 3, 5, 10, 5, 7, 6, 9, 7, 9, 1, 10], [10, 3, 8, 0, 8, 7, 3, 9, 10, 4, 5, 8, 6, 7, 8], [2, 3, 0, 3, 10, 8, 1, 1, 3, 10, 8, 0, 0, 6, 8]]
    # input = [[0.43045255, 0.78681387, 0.07514408, 0.72583933, 0.52916145, 0.87483212, 0.34701621],
    #     [0.68704291, 0.45392742, 0.46862110, 0.67669006, 0.23817468, 0.87520581, 0.67311418],
    #     [0.38505150, 0.05974168, 0.11388629, 0.28978058, 0.66089373, 0.92592403, 0.70718757],
    #     [0.24975701, 0.16937649, 0.42003672, 0.88231235, 0.74635725, 0.59854858, 0.88631100],
    #     [0.64895582, 0.58909596, 0.99772334, 0.85522575, 0.33916707, 0.72873479, 0.26826203],
    #     [0.47939038, 0.88484586, 0.05122520, 0.83527995, 0.37219939, 0.20375257, 0.50482283],
    #     [0.58926554, 0.45176739, 0.25217475, 0.83548120, 0.41687026, 0.00293049, 0.23939052]]
    avg_time = 0
    for test in range(10):
        start = time.time()
        best_chromosome, best_cost = BF(input)

        append = 0 
        end = time.time()
        iter_time = end - start
        avg_time += iter_time
    print("best chromosome", best_chromosome)
    print("Brute Force best cost", best_cost)
    print("Brute Force avg time",avg_time/10)
            