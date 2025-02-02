#coding=utf-8
import math
from multiprocessing import freeze_support
from re import L
import time
import traceback
from common_function import *
from global_config import *
from read_cnf import *

from dwave.system import DWaveSampler, FixedEmbeddingComposite
import minorminer
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridDQMSampler   #real qpu
from dimod.reference.samplers import ExactSolver
import neal
import dimod
from collections import defaultdict
import json

from pysat.solvers import Glucose3
# import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading


def reoraginize(clauses):
    literals = [
        literal
        for clause in clauses
            for literal in clause
    ]
    literals = set(literals)
    variables = set([Var(literal) for literal in literals])
    return literals, variables

# def split(clauses_number, clauses):
#     variables = []
#     for clause in clauses:
#         temp_vars = [Var(literal) for literal in clauses]
#         for temp_var in temp_vars:
#             if temp_var in variables

#     return temp_vars


# clauses, literals = readCNF_intr('/Users/siwei/workspace/solveClauses128/uuf200-048_8.txt')

def solve(lines, is_real):
    # clauses, literals = readCNF(path)
    # print(lines)
    # 判断是否使用真机
    lines = lines[0]
    is_real = is_real[0]

    is_real = True if is_real == "True" else False

    clauses, literals = dealCNF_intr(lines)
    # print(clauses)
    # if len(clauses) < 30:  #or len(clauses) > 300
    #     return None

    # print(clauses)
    # variables = set([Var(literal) for ÷literal in literals])
    # clauses = clauses[:155]
    literals, variables = reoraginize(clauses)

    satisfy, _ = solveSAT(clauses)
    # print(satisfy, result)
    # exit()
    # bqm = dimod.BQM.from_qubo(Q)
    clause2Q = {clause:getQ(clause) for clause in clauses}

    decompose_clauses = set([
        subclause
        for clause in clauses
        for subclause in decompose(clause)
    ])
    clause2Q = {
        clause:getQ2(clause)
        for clause in decompose_clauses
    }

    def getEdge2coef_clause(clause2Q):
        edge2coef_clause = defaultdict(list)
        bias = 0
        for clause, Q in clause2Q.items():
            for edge, coef in Q.items():
                if edge == 'bias':
                    bias += coef
                    continue
                edge2coef_clause[edge].append((coef, clause))  #应该是对sub-clauses

        return edge2coef_clause, bias
    edge2coef_clause, bias = getEdge2coef_clause(clause2Q)

    def converge(edge2coef_clause):
        coef_edge_clause = []
        for edge, coef_clauses in edge2coef_clause.items():
            sum_coef = 0
            for coef, clause in coef_clauses:
                sum_coef += coef

            if edge[0] == edge[1]:
                abs_coef = abs(sum_coef) / 2
            else:
                abs_coef = abs(sum_coef)
            coef_edge_clause.append((sum_coef, abs_coef, edge, coef_clauses))

        rank_coefs = [elm[1] for elm in coef_edge_clause]
        rank_coefs = list(set(rank_coefs))

        coef_edge_clause.sort(key=lambda elm: elm[1], reverse=True)
        rank_coefs.sort(reverse=True)

        return rank_coefs, coef_edge_clause

    rank_coefs, coef_edge_clause = converge(edge2coef_clause)

    # clause2max_coef = defaultdict(float)
    # for sum_coef, abs_coef, edge, coef_clauses in coef_edge_clause:
    #     for coef, clause in coef_clauses:
    #         clause2max_coef[clause] = max([clause2max_coef[clause], abs_coef])

    # for clause, max_coef in clause2max_coef.items():
    #     if max_coef == 0:
    #         clause2max_coef[clause] = rank_coefs[0]

    # clause2Q = {
    #     clause: {
    #         item: coef * (rank_coefs[0] / clause2max_coef[clause])
    #         for item, coef in Q.items()
    #     }
    #     for clause, Q in clause2Q.items()
    # }

    # edge2coef_clause, bias = getEdge2coef_clause(clause2Q)
    # rank_coefs2, coef_edge_clause = converge(edge2coef_clause)

    Q = {
        elm[2]: elm[0]  #/rank_coefs[0]
        for elm in coef_edge_clause
    }

    Q, bias = normalizeQ(Q, bias)

    # Q = {('s1', 's2'): 0.5, ('s1', 's3'): 0.5, ('s2', 's3'): 0.5}
    # qpu = DWaveSampler(solver={'topology__type': 'chimera'})  #2000Q
    # embedding = minorminer.find_embedding(Q.keys(), qpu.edgelist)
    # sampler = FixedEmbeddingComposite(qpu, embedding)

    # embedding = minorminer.find_embedding(source_edgelist, target_edgelist,
    #                                         **embedding_parameters)
    # results = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=10)


    bqm = dimod.BQM.from_qubo(Q)
    if not is_real:
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=1)
    else:
        print('embedding')
        print('before sample')
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample(bqm, num_reads=2)
        print('after sample')

    # print(bias)
    # print(sampleset)
    # print(sampleset.first.energy)
    result = sampleset.first

    var_result = dealResult(result.sample)
    if satisfy:
        var_result['satisfy'] = "1"
    else:
        var_result['satisfy'] = "0"

    if isSatisfy(result.sample, clauses):
        var_result['isSatisfy'] = "1"
    else:
        var_result['isSatisfy'] = "0"
    # print(var_result)
    return var_result #int(satisfy), int(isSatisfy(result.sample, clauses)),

def dealResult(model):
    result = {}
    for item, v in model.items():
        if type(item) == int:
            result[str(item)] = str(v)
    return result

# 如何gap 证明增加了

# result = solve(cnf_path)
# print(result)



def readResult(path):
    Y_sat = []
    Y_unsat = []

    caled_cnfs = []
    rf = open(path, 'r', encoding='utf8')
    rows = rf.read().strip('\n').split('\n')[1:]
    rf.close()
    for row in rows:
        satisfy, result_satisfy, energy, var_num, clause_num, clause_name = row.split(',')


        satisfy = satisfy == 'True'
        energy = float(energy)
        var_num = int(var_num)
        clause_num = int(clause_num)

        
        if not satisfy and clause_num < 300:
            continue

        caled_cnfs.append(clause_name)
        # energy /= math.sqrt(clause_num)
        if satisfy:
            Y_sat.append(energy)
        else:
            Y_unsat.append(energy)

    return Y_sat, Y_unsat, caled_cnfs
