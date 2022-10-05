import numpy as np
import os, sys
import random
 
append = 0

# input = [
#     [10, 20, 23,  4],
#     [15, 13,  6, 25],
#     [ 2, 22, 53, 34],
#     [12,  3, 14, 17]
#     ]
input = [[0.43045255, 0.78681387, 0.07514408, 0.72583933, 0.52916145, 0.87483212, 0.34701621],
    [0.68704291, 0.45392742, 0.46862110, 0.67669006, 0.23817468, 0.87520581, 0.67311418],
    [0.38505150, 0.05974168, 0.11388629, 0.28978058, 0.66089373, 0.92592403, 0.70718757],
    [0.24975701, 0.16937649, 0.42003672, 0.88231235, 0.74635725, 0.59854858, 0.88631100],
    [0.64895582, 0.58909596, 0.99772334, 0.85522575, 0.33916707, 0.72873479, 0.26826203],
    [0.47939038, 0.88484586, 0.05122520, 0.83527995, 0.37219939, 0.20375257, 0.50482283],
    [0.58926554, 0.45176739, 0.25217475, 0.83548120, 0.41687026, 0.00293049, 0.23939052]]
    
def permutation(ele, l ,r):
    global append
    if l==r:
        print("append", ele)
        g = ele.copy()
        ans[append] = g
        print(append,ans[append])
        append+=1

        #print("ans", ans)
    else:
        for i in range(l ,r+1):
            ele[l], ele[i] = ele[i], ele[l]
            permutation(ele, l+1, r)

            ele[l], ele[i] = ele[i] , ele[l]
            

def cal_cost(ans):
    
    best_cost = sys.maxsize

    for i, a in enumerate(ans):
        print("a",a)
        cost = 0
        print(a)
        for i, aa in enumerate(a):
            cost += input[i][aa]
        print("cost", cost)
        if cost < best_cost:
            best_cost = cost
            best_chromosome = a
    print("best_cost", best_cost)
    return best_chromosome, best_cost

 
len_of_gene = len(input)
total = 1
for i in range(1, len_of_gene+1):
    total *= i
ans = [0]*total

indexs = np.arange(len_of_gene)
#print(indexs)
permutation(indexs, 0, len_of_gene-1)

best_chromosome, best_cost = cal_cost(ans)
print("best chromosome", best_chromosome)