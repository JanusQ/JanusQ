#coding=utf-8
from collections import defaultdict
from itertools import product
import logging
import random
from pysat.solvers import Glucose3
from sympy import appellf1
import neal
import dimod
import time

def solveSAT(clauses):
    g = Glucose3()
    for cluase in clauses:
        g.add_clause(cluase)
    return g.solve(), g.get_model()
    # print(g.solve())
    # print(g.get_model())

def index2Chimera(index):  # 把我定义的index转换为标准的index
    return index//4, int(index%4)


def isVar(obj):
    return isinstance(obj, int)

def isAux(obj):
    return isinstance(obj, tuple)

def Var(lit):
    return abs(int(lit))

def decompose(clause):
    clause = list(clause)
    clause.sort(key=lambda elm: Var(elm))  #要改成activity
    
    if len(clause) == 2:
        return [tuple(('OR', clause[0], clause[1]))]
    elif len(clause) == 3:
        aux = (clause[0], clause[1])
        return [
            ('OR', aux, clause[2]),
            ('EQUAL', aux, clause[0], clause[1])
        ]
    logging.exception('the number of clauses is incorrect.')
    # raise Exception('没考虑这种情况')

# 适配decompose的
def getQ2(clause):
    Q = defaultdict(float)
    Q['bias'] = 0

    if clause[0] == 'OR':
        clause = [clause[1], clause[2]]

        if isAux(clause[0]):
            var1, var2 = clause[0], abs(clause[1])
            positive1, positive2 = True, clause[1] > 0
        else:
            var1, var2 = abs(clause[0]), abs(clause[1])
            positive1, positive2 = clause[0] > 0, clause[1] > 0

        if positive1 and positive2:
            # 1 - a - b + ab
            Q['bias'] += 1
            Q[(var1, var1)] -= 1
            Q[(var2, var2)] -= 1
            Q[(var1, var2)] += 1
        elif positive1 and not positive2:
            # b - ab
            Q[(var2, var2)] += 1
            Q[(var1, var2)] -= 1
        elif not positive1 and positive2:
            # a - ab
            Q[(var1, var1)] += 1
            Q[(var1, var2)] -= 1
        elif not positive1 and not positive2:
            # ab
            Q[(var1, var2)] += 1

    elif clause[0] == 'EQUAL':
        aux = clause[1]
        var1, var2 = abs(clause[2]), abs(clause[3])
        positive1, positive2 = clause[2] > 0, clause[3] > 0

        # aux <-> a | b
        if positive1 and positive2:
            # aux + a + b - 2aux a - 2 aux b + ab
            Q[(aux, aux)] += 1
            Q[(var1, var1)] += 1
            Q[(var2, var2)] += 1
            Q[(aux, var1)] -= 2
            Q[(aux, var2)] -= 2
            Q[(var1, var2)] += 1
        elif positive1 and not positive2:
            # 1 - aux + 2 a - b - 2aux a + 2 aux b - ab
            Q['bias'] += 1
            Q[(aux, aux)] -= 1
            Q[(var1, var1)] += 2
            Q[(var2, var2)] -= 1
            Q[(aux, var1)] -= 2
            Q[(aux, var2)] += 2
            Q[(var1, var2)] -= 1
        elif not positive1 and positive2:
            # 1 - aux - a + 2 b + 2aux a - 2 aux b - ab
            Q['bias'] += 1
            Q[(aux, aux)] -= 1
            Q[(var1, var1)] -= 1
            Q[(var2, var2)] += 2
            Q[(aux, var1)] += 2
            Q[(aux, var2)] -= 2
            Q[(var1, var2)] -= 1
        elif not positive1 and not positive2:
            # 3 - 3aux - 2a - 2b  + 2aux a + 2 aux b + ab
            Q['bias'] += 3
            Q[(aux, aux)] -= 3
            Q[(var1, var1)] -= 2
            Q[(var2, var2)] -= 2
            Q[(aux, var1)] += 2
            Q[(aux, var2)] += 2
            Q[(var1, var2)] += 1

    return Q

def getQ(clause):
    Q = defaultdict(float)

    # 假设clause已经sort过了
    clause = list(clause)
    clause.sort(key=lambda elm: Var(elm))  #要改成activity

    Q['bias'] = 0
    def addClause(clause):
        if len(clause) == 0:
            pass
        elif len(clause) == 1:
            var = abs(clause[0])
            positive = clause[0] > 0
            if positive:
                Q[(var, var)] -= 1
                Q['bias'] += 1
            else:
                Q[(var, var)] += 1
        elif len(clause) == 2:
            if isAux(clause[0]):
                var1, var2 = clause[0], abs(clause[1])
                positive1, positive2 = True, clause[1] > 0
            else:
                var1, var2 = abs(clause[0]), abs(clause[1])
                positive1, positive2 = clause[0] > 0, clause[1] > 0

            if positive1 and positive2:
                # 1 - a - b + ab
                Q['bias'] += 1
                Q[(var1, var1)] -= 1
                Q[(var2, var2)] -= 1
                Q[(var1, var2)] += 1
            elif positive1 and not positive2:
                # b - ab
                Q[(var2, var2)] += 1
                Q[(var1, var2)] -= 1
            elif not positive1 and positive2:
                # a - ab
                Q[(var1, var1)] += 1
                Q[(var1, var2)] -= 1
            elif not positive1 and not positive2:
                # ab
                Q[(var1, var2)] += 1

        elif len(clause) == 3:
            var1, var2 = abs(clause[0]), abs(clause[1])
            positive1, positive2 = clause[0] > 0, clause[1] > 0

            aux = (clause[0], clause[1])
            # aux <-> a | b
            if positive1 and positive2:
                # aux + a + b - 2aux a - 2 aux b + ab
                Q[(aux, aux)] += 1
                Q[(var1, var1)] += 1
                Q[(var2, var2)] += 1
                Q[(aux, var1)] -= 2
                Q[(aux, var2)] -= 2
                Q[(var1, var2)] += 1
            elif positive1 and not positive2:
                # 1 - aux + 2 a - b - 2aux a + 2 aux b - ab
                Q['bias'] += 1
                Q[(aux, aux)] -= 1
                Q[(var1, var1)] += 2
                Q[(var2, var2)] -= 1
                Q[(aux, var1)] -= 2
                Q[(aux, var2)] += 2
                Q[(var1, var2)] -= 1
            elif not positive1 and positive2:
                # 1 - aux - a + 2 b + 2aux a - 2 aux b - ab
                Q['bias'] += 1
                Q[(aux, aux)] -= 1
                Q[(var1, var1)] -= 1
                Q[(var2, var2)] += 2
                Q[(aux, var1)] += 2
                Q[(aux, var2)] -= 2
                Q[(var1, var2)] -= 1
            elif not positive1 and not positive2:
                # 3 - 3aux - 2a - 2b  + 2aux a + 2 aux b + ab
                Q['bias'] += 3
                Q[(aux, aux)] -= 3
                Q[(var1, var1)] -= 2
                Q[(var2, var2)] -= 2
                Q[(aux, var1)] += 2
                Q[(aux, var2)] += 2
                Q[(var1, var2)] += 1

            addClause([aux, clause[2]])
        elif len(clause) > 3:
            logging.exception(clause, 'is not longer than expected')

    addClause(clause)    
    return dict(Q)


def normalizeQ(Q, offset):
    max_coef = 0
    for item, coef in Q.items():
        coef = abs(coef)
        if item[0] != item[1]:
            max_coef = max([max_coef, coef])
        else:
            max_coef = max([max_coef, coef/2])

    temp_Q = defaultdict(float)

    for elm, coef in Q.items():
        temp_Q[elm] = coef/max_coef

    return temp_Q, offset/max_coef

def isSatisfy(model, clauses):
    for clause in clauses:
        is_satisfy = False
        for literal in clause:
            var = Var(literal)
            if (literal < 0 and model[var] == 0) or (literal > 0 and model[var] == 1):
                is_satisfy = True
                break

        if not is_satisfy:
            return False
    return True

def resetQ(Q):
    newQ = dict()
    for k,v in Q.items():
        if k == 'bias':
            continue
        newQ[k] = v

    return newQ

def scan(Q, clauses, basic_var_num):
    
    all_scores = []
    # print(Q)
    vars = [item[0] for item in Q if item[0] == item[1] ]
    basic_vars = [item[0] for item in Q if item[0] == item[1] and type(item[0])==int]
    # vars = [item[0] for item in Q if item[0] == item[1] and type(item[0])==int] #todo:要问一下是不是要改
    var2index = {}
    var_num = len(vars)
    for i,var in enumerate(vars):
        var2index[var] = i

    solution_space = []
    satisfy_combination = []

    newQ = resetQ(Q)
    bqm = dimod.BQM.from_qubo(newQ)
    results = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=5)

    for sample, energy in results.data(['sample', 'energy']):
        combination = []
        for var in vars:
            combination.append(sample[var])
        satisfy_combination.append(combination)

    # for i in range(0,10):
    #     satisfy, res = solveSAT(clauses)
    #     combination = []
    #     if satisfy:
    #         res.sort(key = lambda x:abs(x))
    #         for var in vars:
    #             if type(var) == int:
    #                 if res[abs(var)-1] > 0:
    #                     combination.append(1)
    #                 else:
    #                     combination.append(0)
    #             else:
    #                 if res[abs(var[0])-1] >0 or res[abs(var[1])-1] > 0:
    #                     combination.append(1)
    #                 else:
    #                     combination.append(0)
    #         satisfy_combination.append(combination)

    if len(satisfy_combination) > 0:
        while len(solution_space) < 400:
            for combination in satisfy_combination:
                # changenum = random.randint(1,5)
                array = [int(basic_var_num*0.5),int(basic_var_num*0.7),int(basic_var_num)*0.8]
                changenum = random.choice(array)
                combination_copy = combination.copy()
                chanage_var = []
                while changenum > 0:
                    var = random.choice(basic_vars)
                    index = var2index[var]
                    # combination_copy[index] = random.choice([0,1])
                    combination_copy[index] = 1 if combination_copy[index]== 0 else 0
                    chanage_var.append(var)
                    changenum -= 1
                for i1 in range(0,len(chanage_var)):
                    for i2 in range(i1+1,len(chanage_var)):
                        if (i1,i2) in vars:
                            combination_copy[var2index[(i1,i2)]] = (combination_copy[var2index[i1]] | combination_copy[var2index[i2]])
                        if (i2,i1) in vars:
                            combination_copy[var2index[(i2,i1)]] = (combination_copy[var2index[i1]] | combination_copy[var2index[i2]])
                solution_space.append(combination_copy)
        
        while len(solution_space) < 500:
            solution_space.append([random.choice([0,1]) for _ in range(var_num)])

    else:
        solution_space = [
        [random.choice([0,1]) for _ in range(var_num)]
        for _ in range(500)
    ]

    # solution_space = product(*[[0,1]]*var_num)
    
    # solution_space = [
    #     [random.choice([0,1]) for _ in range(var_num)]
    #     for _ in range(1000)
    # ]
    
    for combination in solution_space: #1000*60*60
        # print(Q)
        score = 0
        for i1 in range(1, var_num+1):
            for i2 in range(i1, var_num+1):
                v1, v2 = combination[i1-1], combination[i2-1]
                var1, var2 =  vars[i1-1], vars[i2-1]
                if var1 != var2:
                    # print('b', i1, i2, Q[(var1, var2)], Q[(var2, var1)]) 
                    score += (Q[(var1, var2)] + Q[(var2, var1)]) * v1 * v2
                else:
                    # print('a', i1, Q[(var1, var1)]) 
                    score += Q[(var1, var1)] * v1

        score += Q['bias']
        combination = {
            var: combination[i]
            for i, var in enumerate(vars)
        }
        satisfy = isSatisfy(combination, clauses)
        all_scores.append((satisfy, score, combination,))
    
    all_scores.sort(key=lambda x: x[1])
    for satisfy, score, combination in all_scores:
        if not satisfy:
            gap = score
            break
    
    return all_scores, gap
    
