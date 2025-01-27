


from  jax import jit
from jax import random
import jax.numpy as jnp
import numpy as np
from typing import Iterable
from jax.example_libraries.optimizers import adam
from jax import value_and_grad
from tqdm import tqdm
from .solver import Solver
from ..predicate import Predicate

import jax.numpy as jnp


def Equal(stateA:jnp.array, stateB:jnp.array):
    
    return jnp.mean(jnp.abs(stateA-stateB))

def sgd_cost_function(parms:jnp.ndarray,*objectivefuncs):
    """ cost function: gradient descent
    Args:
        parms (jnp.ndarray): the parameters to be optimized
        objectivefuncs (Iterable): the objective functions to be optimized
    """
    return  jnp.sum(jnp.array([objectivefunc(parms) for objectivefunc in objectivefuncs]))

def build_density(parms,samples):
    """ 重建量子态
    Args:
        parms: 重建参数
        outputs: 重建的量子态
    return:
        build_state: 重建的量子态
    """
    return jnp.sum(jnp.array([parm*base for parm,base in zip(parms,samples)]),axis=0)

class SGDSolver(Solver):
    def __init__(self, **config):
        super().__init__(**config)
        self.paramsize = 0
        self.tracepoint_states= {}
        self.strongest_weight = config['strongest_weight']
        self.max_weight = config['max_weight']
        self.high_weight = config['high_weight']
        self.min_weight = config['min_weight']
        self.step_size = config['step_size']
        self.steps = config['steps']
        self.early_stopping_iter = config['early_stopping_iter']
        self.objectivefuncs = []
    
        
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

    def add_constraint(self,tracepoint_idxes: Iterable[int],predicate: Predicate,args):
        self.objectivefuncs.append(lambda x: self.max_weight*predicate(*[self.tracepoint_states[tracepoint][0](x) for tracepoint in tracepoint_idxes],*args))
    def add_objective(self,tracepoint_idxes: Iterable[int],predicate: Predicate,args):
        self.objectivefuncs.append(lambda x: predicate(*[self.tracepoint_states[tracepoint][0](x) for tracepoint in tracepoint_idxes],*args))
        self.objectivefun = self.objectivefuncs[-1]
    def train_params(self,objective:callable,parms:Iterable):
        min_iter = 0
        opt_init, opt_update, get_params = adam(self.step_size)
        min_cost = 1e10
        last_cost = min_cost
        opt_state = opt_init(parms)
        with tqdm(range(self.steps)) as pbar:
            for i in pbar:
                params = get_params(opt_state)
                cost, grads = value_and_grad(objective)(params)
                opt_state=  opt_update(i, grads, opt_state)
                if cost < min_cost:
                    min_cost = cost
                    bast_params = params
                    pbar.set_description(f'sgd optimizing')
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                if jnp.abs(min_cost-last_cost) < 1e-5:
                    min_iter += 1
                else:
                    min_iter = 0
                ## when the loss function changes less than 1e-5 for 50 consecutive iterations, it is considered to converge
                if min_iter > self.early_stopping_iter:
                    pbar.set_description(f'sgd optimizing converge')
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                    break
                last_cost = min_cost
        return bast_params,min_cost

    def solve(self):
        equal_states = [self.tracepoint_states[tracepointidx] for tracepointidx in self.tracepoint_states if len(self.tracepoint_states[tracepointidx])==2]
        for tracepoint in equal_states:
            self.objectivefuncs.append(lambda x: self.strongest_weight*Equal(tracepoint[0](x),tracepoint[1](x)))
        # @jit
        def objective(parms:jnp.ndarray):
            """ cost function: gradient descent
            损失函数： 梯度下降法
            Args:
                parms (jnp.ndarray): the parameters to be optimized
                objectivefuncs (Iterable): the objective functions to be optimized
            """
            return  jnp.sum(jnp.array([objectivefunc(parms) for objectivefunc in self.objectivefuncs]))
        self.paramspace= random.normal(random.PRNGKey(np.random.randint(0,1000)), (self.paramsize,))
        opt_params,opt_objective = self.train_params(objective,self.paramspace)
        opt_input_state = self.tracepoint_states[0][0](opt_params)
        objective_value = self.objectivefun(opt_params)
        constrain_satisfid = [objectivefunc(opt_params) for objectivefunc in self.objectivefuncs[:-1]]
        return {
            'optimal_input_state': opt_input_state,
            "optimal_gurrantee_value": objective_value,
            "is_assume_satisfied" : constrain_satisfid
        }
