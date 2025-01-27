import gurobipy as gp
from gurobipy import GRB
import numpy as np

def build_RI_state(parms,states):
    return sum(parm * state.real for parm,state in zip(parms,states)),\
        sum(parm * state.imag for parm,state in zip(parms,states))

class LinearSolver(gp.Model):
    def __init__(self,name='linear_program'):
        super().__init__()
        pass

    
    def linear_cost_function_state(self,parms,states,real_state,target='output'):
        """ 损失函数： 量子态的重建误差
        Args:
            parms: 参数
            inputs: 输入的量子态
            outputs: 输出的量子态
            real_state: 真实的量子态
            target: 优化的目标，'input'或者'output'
        """
        buildstate= build_RI_state(parms,states)
        if target == 'input':
            self.setObjective(sum(np.square(buildstate-real_state)))
    
    def solve(self,states,real_target):
        """ 求解线性规划
        Args:
            inputs: 输入的量子态
            outputs: 输出的量子态
            real_target: 真实的量子态或者密度矩阵
            target: 优化的目标，'input'或者'output'
        """
        parms = self.addMVar(shape=len(states),lb=-1,ub=1,name='parms',vtype=GRB.COMPLEX)
        ## add constraints
        self.Params.NonConvex = 2
        self.Params.timeLimit = 6000
        self.Params.BarConvTol = 1e-1
        if len(states[0].shape) == 1:
            self.linear_cost_function_state(parms,states,real_target)
        else:
            self.linear_cost_function_density(parms,states,real_target)
        self.optimize()
        if self.status == GRB.OPTIMAL:
            return [parm.x for parm in parms]
        else:
            print('No solution')
            return None,None
        




import jax.numpy as jnp
from  jax import jit
from jax import random
from jax.example_libraries.optimizers import adam
import numpy as np
from jax import value_and_grad
from tqdm import tqdm


@jit
def sgd_cost_function(parms,states,real_state):
    """ 损失函数： 量子态的重建误差
    Args:
        parms: 重建参数
        inputs: 输入的量子态
        outputs: 重建的量子态
        real_state: 真实的量子态
        target: 重建的目标，input or output
    return:
        cost: 重建误差
    """
    buildinput = build_density(parms,states)
    return  jnp.mean(jnp.abs(buildinput-real_state))

def build_density(parms,outputs):
    """ 重建量子态
    Args:
        parms: 重建参数
        outputs: 重建的量子态
    return:
        build_state: 重建的量子态
    """
    return jnp.sum(jnp.array([parm*output for parm,output in zip(parms,outputs)]),axis=0)


class SgdOptimizer:
    def __init__(self,step_size=0.01,steps=5000):
        self.step_size = step_size
        self.steps = steps
    def optimize_params(self,states,real_target):
        """ 优化器
        Args:
            inputs: 输入的量子态
            outputs: 重建的量子态
            real_target: 真实的量子态
            target: 重建的目标,input or output
        return:
            bast_params: 优化后的参数
        """
        parms= random.normal(random.PRNGKey(np.random.randint(0,1000)), (len(states),))
        min_iter = 0
        opt_init, opt_update, get_params = adam(self.step_size)
        min_cost = 1e10
        last_cost = min_cost
        opt_state = opt_init(parms)

        with tqdm(range(self.steps)) as pbar:
            for i in pbar:
                params = get_params(opt_state)
                cost, grads = value_and_grad(sgd_cost_function)(params,states, real_target)
                opt_state=  opt_update(i, grads, opt_state)
                if cost < min_cost:
                    min_cost = cost
                    bast_params = params
                    pbar.set_description(f'sgd optimizing')
                    #设置进度条右边显示的信息
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                if jnp.abs(min_cost-last_cost) < 1e-5:
                    min_iter+=1
                else:
                    min_iter = 0
                # 当连续50次迭代损失函数变化小于1e-5时，认为收敛
                if min_iter > 50:
                    pbar.set_description(f'sgd optimizing converge')
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                    break
                last_cost = min_cost
        return bast_params