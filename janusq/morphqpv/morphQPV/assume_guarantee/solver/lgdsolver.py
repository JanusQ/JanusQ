

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
class State:
    def __init__(self,value: jnp.array) -> None:
        self.value = value

class StateVector(State):
    def __init__(self,value: jnp.array) -> None:
        assert len(value.shape) == 1, 'The dimension of statevector must be 1'
        super().__init__(value)
    def __expectation__(self, operator: object) -> object:
        return self._to_density_matrix().__expectation__(operator)
    def __trace__(self) -> object:
        return self._to_density_matrix().__trace__()
    def __eq__(self, state: object) -> bool:
        return jnp.mean(jnp.abs(self.value-state.value))
    def __ispure__(self) -> bool:
        return jnp.abs(self.value @ self.value.conj().T - 1)
    def __isvalid__(self) -> bool:
        return self.__ispure__()
    def _to_density_matrix(self):
        return DensityMatrix(jnp.outer(self.value,self.value.conj()))

class DensityMatrix(State):
    def __init__(self,value: jnp.array) -> None:
        assert len(value.shape) == 2, 'The dimension of density matrix must be 2'
        super().__init__(value)
    def __expectation__(self, operator: object) -> object:
        return jnp.trace(self.value @ operator.value)
    def __trace__(self) -> object:
        return jnp.trace(self.value)
    def __eq__(self, state: object) -> bool:
        return jnp.mean(jnp.abs(self.value-state.value))
    def __ispure__(self) -> bool:
        return (jnp.mean(jnp.abs(self.value-self.value@self.value)) + jnp.abs(jnp.trace(self.value)-1))
    def __isvalid__(self) -> bool:
        return jnp.abs(jnp.trace(self.value)-1)

class Predicate:
    ''' A constraint '''
    def __init__(self):
        pass
    def __call__(self, state: State):
        # return self.constraint(state)
        pass
    def is_satisfy(self, state: State): 
        return self.constraint(state)


def Equal(stateA:jnp.array, stateB:jnp.array):
    
    return jnp.mean(jnp.abs(stateA-stateB))

def Expectation(state: jnp.array, operator: object):
    return state.__expectation__(operator)

def NotEqual(stateA:jnp.array, stateB:jnp.array):
    return - jnp.mean(jnp.abs(stateA-stateB))

def Orthogonal(stateA:jnp.array, stateB:jnp.array):
    if len(stateA.shape) == 1:
        return jnp.abs(jnp.dot(stateA,stateB))
    else:
        return jnp.abs(jnp.trace(jnp.dot(stateA,stateB.conj().T)))
    
def Trace(state: jnp.array):
    if len(state.shape) == 1:
        return StateVector(state).__ispure__()
    else:
        return DensityMatrix(state).__ispure__()

def isPure(state: jnp.array):
    if len(state.shape) == 1:
        return StateVector(state).__ispure__()
    else:
        return DensityMatrix(state).__ispure__()

def Expectation(state: jnp.array, operator: object):
    return state.__expectation__(operator)


def wrap_step_size_fun(step_size,method:str= 'constant'):
    def constant_step_size(i):
        return step_size
    if method == 'constant':
        return constant_step_size
    else:
        raise ValueError('currently not support')


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

def adagrad(step_size, momentum=0.9):
    """Construct optimizer triple for Adagrad.

    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    Args:
        step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.
        momentum: optional, a positive scalar value for momentum

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    def init(x0):
        g_sq = jnp.zeros_like(x0)
        m = jnp.zeros_like(x0)
        return x0, g_sq, m

    def update(i, g, state):
        x, g_sq, m = state
        g_sq += jnp.square(g)
        g_sq_inv_sqrt = jnp.where(g_sq > 0, 1. / jnp.sqrt(g_sq), 0.0)
        m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
        x = x - step_size(i) * m
        return x, g_sq, m

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params

class LGDSolver(Solver):
    def __init__(self, **config):
        super().__init__(**config)
        self.step_size = config['step_size']
        self.steps = config['steps']
        self.paramsize = 0
        self.tracepoint_states= {}
        self.objectivefuncs = []
        self.lagrange_constraintfuncs = []
        self.ktt_constraintfuncs = []
        self.lagrange_constraints = []
        self.ktt_constraints = []
    
        
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

    def add_constraint(self, tracepoint_idxes: Iterable[int],predicate: Predicate,args):
        if predicate.Literal == 'Equal' or predicate.Literal == 'IsPure':
            self.add_equal_constraint(tracepoint_idxes,predicate,args)
        else:
            self.add_le_constraint(tracepoint_idxes,predicate,args)

    def add_equal_constraint(self,tracepoint_idxes: Iterable[int],predicate: Predicate,args):
        self.lagrange_constraints.append(lambda x: predicate(*[self.tracepoint_states[tracepoint][0](x) for tracepoint in tracepoint_idxes],*args))
    
    def add_le_constraint(self,tracepoint_idxes: Iterable[int],predicate: Predicate,args):
        self.ktt_constraints.append(lambda x: predicate(*[self.tracepoint_states[tracepoint][0](x) for tracepoint in tracepoint_idxes],*args))

    def add_objective(self,tracepoint_idxes: Iterable[int],predicate: Predicate,args):
        self.objectivefuncs.append(lambda x: predicate(*[self.tracepoint_states[tracepoint][0](x) for tracepoint in tracepoint_idxes],*args))


    def train_params(self,objective:callable,parms:Iterable,lagrangeparamsize:int,kttparamsize:int):
        """ 训练参数
        Args:
            objective (callable): the objective function to be optimized
            parms (Iterable): the parameters to be optimized
            lagrangeparamsize (int): the lagrange parameters size
        Returns:
            bast_params: the optimal parameters
            min_cost: the minimal cost
        """
        min_iter = 0
        opt_init, opt_update, get_params = adagrad(wrap_step_size_fun(self.step_size))
        min_cost = 1e10
        last_cost = min_cost
        opt_state = opt_init(parms)
        with tqdm(range(self.steps)) as pbar:
            for i in pbar:
                params = get_params(opt_state)
                cost, grads = value_and_grad(objective)(params)
                params, g_sq, m =  opt_update(i, grads, opt_state)
                all_params_size = len(parms)
                ## for lagrange multipliers,we use other way to update
                origin_params,lagrange_params,ktt_params= params[:all_params_size-lagrangeparamsize-kttparamsize],params[all_params_size-lagrangeparamsize-kttparamsize:all_params_size-kttparamsize],params[all_params_size-kttparamsize:]
                if kttparamsize > 0:
                    for i,constraintfunc in enumerate(self.ktt_constraintfuncs):
                        ktt_params = ktt_params.at[i].set(jnp.maximum(ktt_params[i]+self.step_size*constraintfunc(params),0))
                if lagrangeparamsize > 0:
                    for i,constraintfunc in enumerate(self.lagrange_constraintfuncs):
                        # lagrange_params = lagrange_params.at[i].set(jnp.maximum(lagrange_params[i]+self.step_size*constraintfunc(params)/(lagrange_params[i]+1e-10),0))
                        lagrange_params = lagrange_params.at[i].set(lagrange_params[i]+self.step_size*constraintfunc(params)/(lagrange_params[i]+1e-10))
                params = jnp.concatenate([origin_params,lagrange_params,ktt_params])
                opt_state = params,g_sq,m
                if cost < min_cost:
                    min_cost = cost
                    bast_params = params
                    pbar.set_description(f'LGD optimizing')
                    #设置进度条右边显示的信息
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                if jnp.abs(min_cost-last_cost) < 1e-5:
                    min_iter += 1
                else:
                    min_iter = 0
                # 当连续50次迭代损失函数变化小于1e-5时，认为收敛
                if min_iter > 50:
                    pbar.set_description(f'sgd optimizing converge')
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                    break
                last_cost = min_cost
        return bast_params,min_cost

    def solve(self):
        equal_states = [self.tracepoint_states[tracepointidx] for tracepointidx in self.tracepoint_states if len(self.tracepoint_states[tracepointidx])==2]
        for tracepoint in equal_states:
            self.lagrange_constraints.append(lambda x: Equal(tracepoint[0](x),tracepoint[1](x)))
        
        lagrangeparamsize = len(self.lagrange_constraints)
        kttparamsize = len(self.ktt_constraints)
        self.all_params_size = self.paramsize + lagrangeparamsize +kttparamsize
        self.paramspace= random.normal(random.PRNGKey(np.random.randint(0,1000)), (self.all_params_size,))
        ## add lagrange coefficient to constraint function
        self.lagrange_constraintfuncs = [lambda x: x[self.paramsize+i]*constraintfunc(x) for i,constraintfunc in enumerate(self.lagrange_constraints)]
        ## add ktt coefficient to constraint function
        self.ktt_constraintfuncs = [lambda x: x[self.paramsize+ lagrangeparamsize+i]*constraintfunc(x) for i,constraintfunc in enumerate(self.ktt_constraints)]
        
        @jit
        def objective(parms:jnp.ndarray):
            """ cost function: gradient descent
            损失函数： 梯度下降法
            Args:
                parms (jnp.ndarray): the parameters to be optimized
                objectivefuncs (Iterable): the objective functions to be optimized
            """
            return  jnp.sum(jnp.array([objectivefunc(parms) for objectivefunc in self.objectivefuncs])) + jnp.sum(jnp.array([constraintfunc(parms) for constraintfunc in self.lagrange_constraintfuncs]))+jnp.sum(jnp.array([constraintfunc(parms) for constraintfunc in self.ktt_constraintfuncs]))
        
        opt_params,opt_objective = self.train_params(objective,self.paramspace,lagrangeparamsize,kttparamsize)
        opt_input_state = self.tracepoint_states[0][0](opt_params)
        objective_value = [objectivefunc(opt_params) for objectivefunc in self.objectivefuncs]
        constraint_value = [constraintfunc(opt_params) for constraintfunc in self.lagrange_constraints]
        
        return {
            'optimal_input_state': opt_input_state,
            "optimal_gurrantee_value": objective_value,
            "constraint_value": constraint_value
        }
