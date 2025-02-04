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


# arg = "-398,424,441;-398,-424,-441;398,424,-441;398,-424,441;397,-398,-424;-397,398;-398,399,401;398,-399;398,-401;396,424,-425;-396,424,425;-396,-424,-425;396,-424,425;-397,424;440,-441,-698,-707;-440,441;441,463,-464;-441,463,464;-441,-463,-464;441,-463,464;-394,395,397;394,-397;399,-400,-698,-708;-399,698;-399,708;-399,400;401,-402,-422;-401,402;-401,422;395,-396,-697,-708;-395,396;354,-385,396;-354,385,396;-354,-385,-396;354,385,-396;425,-697,-708;-425,697;-425,708;-439,440,442;439,-440;-440,698;-440,707;423,-698,-708;-423,698;505,-698,-706;-505,698;-464,698;464,-698,-707;481,-482,-698,-706;-481,698;-358,698;361,-698,-699,-709,-710;-361,698;358,-697,-698,-709,-710;-381,698;377,-698,-710;-377,698;381,-698,-709;628,-698,-703;-628,698;604,-605,-698,-703;-604,698;698,-722;-698,722;546,-698,-705;-546,698;522,-523,-698,-705;-522,698;587,-698,-704;-587,698;563,-564,-698,-704;-563,698;444,-445,-699,-707;-444,707;436,-437,-697,-707;-436,707;455,-456,-702,-707;-455,707;458,-701,-707;448,-449,-700,-707;-448,707;452,-453,-701,-707;-452,707;432,-433,-696,-707;-432,707;-464,707;466,-697,-707;-466,707;468,-696,-707;-468,707;-458,707;460,-700,-707;-460,707;462,-699,-707;-462,707;470,-695,-707;-470,707;-347,707;347,-348,-695,-707;669,-702,-707;-669,707;707,-715;-707,715;442,-443,-463;-442,463;443,-463,486;-443,463,486;-443,-463,-486;443,463,-486;394,-426,437;-394,426,437;-394,-426,-437;394,426,-437;394,-395;393,-394,-426;-393,394;-395,697;-395,708;400,422,-423;-400,422,423;-400,-422,-423;400,-422,423;357,-382,400;-357,382,400;-357,-382,-400;357,382,-400;-417,708;419,-700,-708;-419,708;421,-699,-708;-421,708;414,-415,-702,-708;-414,708;417,-701,-708;429,-695,-708;-429,708;-423,708;427,-696,-708;-427,708;349,-350,-695,-708;-349,708;390,-391,-696,-708;-390,708;407,-408,-700,-708;-407,708;411,-412,-701,-708;-411,708;403,-404,-699,-708;-403,708;-673,708;673,-702,-708;-708,716;708,-716;402,-422,445;-402,422,445;-402,-422,-445;402,422,-445;-402,403,405;402,-403;402,-405;-436,697;-466,697;477,-478,-697,-706;-477,697;-358,697;355,-696,-697,-709,-710;-355,697;384,-697,-709;-384,697;380,-697,-710;-380,697;630,-697,-703;-630,697;-697,721;697,-721;548,-697,-705;-548,697;507,-697,-706;-507,697;518,-519,-697,-705;-518,697;600,-601,-697,-703;-600,697;589,-697,-704;-589,697;-559,697;559,-560,-697,-704;353,-354,-385;-353,354;-354,355,356;354,-355;354,-356;-353,385;385,386,-387;385,-386,387;-385,-386,-387;-385,386,387;438,-439,-465;-438,439;439,-442;-439,465,482;-439,-465,-482;439,465,-482;439,-465,482;-442,443;482,504,-505;-482,504,505;-482,-504,-505;482,-504,505;-505,706;489,-490,-700,-706;-489,706;493,-494,-701,-706;"
# print(solve(arg))