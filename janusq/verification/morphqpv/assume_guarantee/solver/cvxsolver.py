import cvxpy as cp
import numpy as np
from .solver import Solver
from typing import Iterable


def Equal(stateA, stateB):
    
    return cp.mean(cp.abs(stateA-stateB))

def Orthogonal(stateA, stateB):
    if len(stateA.shape) == 1:
        return cp.abs(cp.dot(stateA,stateB))
    else:
        return cp.abs(cp.trace(cp.dot(stateA,stateB.conj().T)))

def Trace(state):
    if len(state.shape) == 1:
        return cp.trace(cp.outer(state,state.conj()))
    else:
        return cp.trace(state)

def Expectation(state, operator: object):
    return cp.trace(state @ operator.value)

def NotEqual(stateA , stateB):
    return cp.mean(cp.abs(stateA-stateB))>0.01

def isPure(state):
    if len(state.shape) == 1:
        return cp.dot(state,state.conj()) == 1
    else:
        return cp.trace(state @ state.conj().T) == 1


def build_density(parms,samples):
    """ 重建量子态
    Args:
        parms: 重建参数
        outputs: 重建的量子态
    return:
        build_state: 重建的量子态
    """
    return cp.sum([parm*base for parm,base in zip(parms,samples)])

class CVXSolver(Solver):
    def __init__(self, **config):
        super().__init__(**config)
        self.paramsize = 0
        self.tracepoint_states= {}
        self.strongest_weight = 1000
        self.max_weight = 100
        self.high_weight = 10

    def add_relation(self, tracepoint1, tracepoint2, relation: tuple):
        inputs,outputs = relation
        assert len(inputs)== len(outputs)
        paramsize = len(inputs)
        params_s_idx= self.paramsize
        self.paramsize+=paramsize
        params_e_idx = self.paramsize
        if tracepoint1 not in self.tracepoint_states:
            self.tracepoint_states[tracepoint1] = []
        if tracepoint2 not in self.tracepoint_states:
            self.tracepoint_states[tracepoint2] = []
        
        self.tracepoint_states[tracepoint1].append(lambda x: build_density(x[params_s_idx:params_e_idx],inputs))
        self.tracepoint_states[tracepoint2].append(lambda x: build_density(x[params_s_idx:params_e_idx],outputs))

    def add_constraint(self,tracepoint_idxes: Iterable[int],predicate: function,args):
        self.constraintfuncs.append(lambda x: predicate(*[self.tracepoint_states[tracepoint][0](x) for tracepoint in tracepoint_idxes],*args))
    def add_objective(self,tracepoint_idxes: Iterable[int],predicate: function,args):
        self.objectivefun = lambda x: predicate(*[self.tracepoint_states[tracepoint][0](x) for tracepoint in tracepoint_idxes],*args)

    def solve(self):
        equal_states = [self.tracepoint_states[tracepointidx] for tracepointidx in self.tracepoint_states if len(self.tracepoint_states[tracepointidx])==2]
        for tracepoint in equal_states:
            self.constraintfuncs.append(lambda x: Equal(tracepoint[0](x),tracepoint[1](x)))

        self.paramspace= cp.Variable(self.paramsize)
        objective = cp.Maximize(self.objectivefun(self.paramspace))
        constraints = [fun(self.paramspace) for fun in self.constraintfuncs ]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        opt_input_state = build_density(self.paramspace.value)
        objective_value = prob.value
        return {
            'optimal_input_state': opt_input_state,
            "optimal_gurrantee_value": objective_value
        }
