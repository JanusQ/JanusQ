import jax.numpy as jnp
from  jax import jit
from jax import random
from jax.example_libraries.optimizers import adam
import numpy as np
from jax import value_and_grad
from tqdm import tqdm

def purity(build_state):
    """ purity of a state """
    ## 保证build_state 是纯态
    if len(build_state.shape)>1: # density matrix
        ## 保证trace=1
        ## 保证 \rho^2 = \rho
        eignvalues = jnp.linalg.eigvals(build_state)
        return 10*(jnp.mean(jnp.abs(build_state -build_state@build_state)) + jnp.abs(jnp.trace(build_state)-1))
    else: # statevector
        ## 保证 <\psi|\psi> = 1
        return jnp.abs(build_state @ build_state.conj().T - 1)
    

@jit
def sgd_input_cost_function(parms,inputs,outputs,real_state):
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
    buildinput = build_density(parms,inputs)
    return  jnp.mean(jnp.abs(buildinput-real_state))
    
@jit
def sgd_output_cost_function(parms,inputs,outputs,real_state):
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
    buildinput = build_density(parms,inputs)
    buildoutput = build_density(parms,outputs)
    if len(buildinput.shape)>1:
        return  jnp.mean(jnp.abs(buildoutput-real_state)) + purity(buildinput)
    else:
        return  jnp.mean(jnp.abs(buildoutput-real_state)) + jnp.abs(buildinput @ buildinput.conj().T - 1)+ jnp.abs(buildoutput @ buildoutput.conj().T - 1)
    # + 100*(jnp.abs(buildinput @ buildinput.conj().T - 1)+ jnp.abs(buildoutput @ buildoutput.conj().T - 1))
    

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
    def optimize_params(self,inputs,outputs,real_target,target='input'):
        """ 优化器
        Args:
            inputs: 输入的量子态
            outputs: 重建的量子态
            real_target: 真实的量子态
            target: 重建的目标，input or output
        return:
            bast_params: 优化后的参数
        """
        parms= random.normal(random.PRNGKey(np.random.randint(0,1000)), (len(inputs),))
        min_iter = 0
        opt_init, opt_update, get_params = adam(self.step_size)
        min_cost = 1e10
        last_cost = min_cost
        opt_state = opt_init(parms)
        if target == 'input':
            sgd_cost_function = sgd_input_cost_function
        elif target == 'output':
            sgd_cost_function = sgd_output_cost_function
        with tqdm(range(self.steps)) as pbar:
            for i in pbar:
                params = get_params(opt_state)
                cost, grads = value_and_grad(sgd_cost_function)(params,inputs, outputs, real_target)
                opt_state=  opt_update(i, grads, opt_state)
                if cost < min_cost:
                    min_cost = cost
                    bast_params = params
                    pbar.set_description(f'sgd optimizing {target}')
                    #设置进度条右边显示的信息
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                if jnp.abs(min_cost-last_cost) < 1e-5:
                    min_iter+=1
                else:
                    min_iter = 0
                # 当连续50次迭代损失函数变化小于1e-5时，认为收敛
                if min_iter > 100:
                    pbar.set_description(f'sgd optimizing {target} converge')
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                    break
                last_cost = min_cost
        return bast_params