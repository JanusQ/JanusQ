import numpy as np

def purity(build_state):
    """ purity of a state """
    ## 保证build_state 是纯态
    if len(build_state.shape)>1: # density matrix
        ## 保证trace=1
        ## 保证 \rho^2 = \rho
        eignvalues = np.linalg.eigvals(build_state)
        return 10*(np.mean(np.abs(build_state -build_state@build_state)) + np.abs(np.trace(build_state)-1))
    else: # statevector
        ## 保证 <\psi|\psi> = 1
        return np.abs(build_state @ build_state.conj().T - 1)
    
def input_cost_function(parms,inputs,outputs,real_state):
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
    return  np.mean(np.abs(buildinput-real_state))
    
def output_cost_function(parms,inputs,outputs,real_state):
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
        return  np.mean(np.abs(buildoutput-real_state)) + purity(buildinput)
    else:
        return  np.mean(np.abs(buildoutput-real_state))+ 10*(np.abs(buildinput @ buildinput.conj().T - 1)+ np.abs(buildoutput @ buildoutput.conj().T - 1))
    

def build_density(parms,outputs):
    """ 重建量子态
    Args:
        parms: 重建参数
        outputs: 重建的量子态
    return:
        build_state: 重建的量子态
    """
    return np.sum(np.array([parm*output for parm,output in zip(parms,outputs)]),axis=0)