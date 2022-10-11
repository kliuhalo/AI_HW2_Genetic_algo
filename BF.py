import numpy as np
import os, sys
import random
import json
import time
 
append = 0
best_cost = sys.maxsize
best_chromosome = None

def permutation(ele, l ,r):
    if l==r:
        g = ele.copy()
        cal_cost2(g)
    else:
        for i in range(l ,r+1):
            ele[l], ele[i] = ele[i], ele[l]
            permutation(ele, l+1, r)
            ele[l], ele[i] = ele[i] , ele[l]

def cal_cost2(a):
    global best_chromosome
    global best_cost
    cost = 0
    for i, aa in enumerate(a):
        cost += input[i][aa]
    if cost < best_cost:
        best_cost = cost
        best_chromosome = a

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

    indexs = np.arange(len_of_gene)
    permutation(indexs, 0, len_of_gene-1)

    return best_chromosome, best_cost
    


if __name__ == '__main__':

    
    with open('input.json', 'r') as inputFile:
        data = json.load(inputFile)
        for key in data:
            input = data[key]
            best_chromosome, best_cost = BF(input)

            print("best chromosome", best_chromosome)
            print("Brute Force best cost", best_cost)

    # start = time.time()
    # BF(input)
    # end = time.time()
    # print("Brute Force avg time",avg_time/10)
    # print("Brute force time", end-start)
            