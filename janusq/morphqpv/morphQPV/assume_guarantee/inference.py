from .sample import process_sample,process_stat_sample

from .optimizer.linear_solver import LinearSolver
from .optimizer.sgd import SgdOptimizer
from .optimizer.SimulatedAnnealing import SimulatedAnnealing
from .optimizer.exact_solver import InnerProductOptimizer
from .optimizer.costfunctions import input_cost_function,output_cost_function
from morphQPV.execute_engine.excute import ExcuteEngine,convert_state_to_density,convert_density_to_state
import numpy as np


def compose(parms,states):
    """ 通过参数和状态构建量子态
    Args:
        parms: 参数
        states: 状态
    Returns:
        build_state: 构建的量子态
    """
    return np.sum(np.array([parm*state for parm, state in zip(parms,states)]),axis=0)

def decompose(inputs,outputs,real_target,target='input',optimizer='quadratic',steps=10000,learning_rate=0.001,jobname='quadratic'):
    """ 优化参数
    Args:
        inputs: 输入的量子态
        outputs: 输出的量子态
        real_target: 真实的目标态
        target: 优化的目标，'input'或者'output'
        optimizer: 优化器，'quadratic'或者'descent'
        steps: 迭代次数
    Returns:
        best_params: 最优参数
    """
    if optimizer == 'exact_solver':
        return InnerProductOptimizer().solve(inputs,outputs,real_target,target=target)
    elif optimizer == 'quadratic':
        model = LinearSolver(name=jobname)
        # model.Params.LogToConsole = 0
        best_params = model.solve(inputs,outputs,real_target,target,jobname=jobname)
        return best_params
    elif optimizer == 'descent':
        model = SgdOptimizer(steps=steps,step_size=learning_rate)
        best_params = model.optimize_params(inputs,outputs,real_target,target)
        return best_params
    elif optimizer == 'annealing':
        model = SimulatedAnnealing()
        if target == 'input':
            cost_function = input_cost_function
        elif target == 'output':
            cost_function = output_cost_function
        best_params = model.optimize_params(len(inputs),cost_function,inputs,outputs,real_target)
        return best_params

def infer_process_by_density(process:list,real_density,output_qubits,method,base_num,target='input',input_qubits=None,optimizer='quadratic',learning_rate=0.001,steps=5000,device = 'simulate'):
    """ 通过密度矩阵构建量子态
    Args:
        process: 量子线路
        real_density: 真实的密度矩阵
        output_qubits: 输出的量子比特
        method: 采样方法
        basenum: 采样基数
        target: 优化的目标，'input'或者'output'
        input_qubits: 输入的量子比特
        optimizer: 优化器，'quadratic'或者'descent'
        learning_rate: 学习率
        steps: 迭代次数
    Returns:
        build_input: 构建的输入量子态
        build_output: 构建的输出量子态
    """
    inputs = []
    outputs = []
    inputs,outputs = process_sample(process,input_qubits,base_num=base_num,method=method,output_qubits=output_qubits,device=device)
    inputs = np.array(list(map(convert_state_to_density, inputs)))
    trained_parms = decompose(inputs,outputs,real_density,target=target,optimizer=optimizer,learning_rate=learning_rate,steps=steps)
    build_input = compose(trained_parms,inputs)
    build_input = build_input/build_input.trace()
    build_output = compose(trained_parms,outputs)
    build_output = build_output/build_output.trace()
    return build_input,build_output

def infer_process_by_statevector(process:list,real_state,method,base_num,input_qubits=None,output_qubits=None,target='input',optimizer='quadratic',learning_rate=0.01,steps=5000,jobname='quadratic',device='simulate'):
    """ 通过量子状态向量构建量子态
    Args:
        process: 量子线路
        real_state: 真实的量子态
        method: 采样方法
        base_num: 采样基数
        input_qubits: 输入的量子比特
        target: 优化的目标，'input'或者'output'
        optimizer: 优化器，'quadratic'或者'descent'
        learning_rate: 学习率
        steps: 迭代次数
    Returns:
        build_input: 构建的输入量子态
        build_output: 构建的输出量子态
    """
    if input_qubits is None:
        input_qubits = ExcuteEngine.get_qubits(process)
    inputs,outputs = process_sample(process,input_qubits,base_num=base_num,method=method,output_qubits=output_qubits,device=device)
    outputs = np.array(list(map(convert_density_to_state, outputs)))
    trained_parms = decompose(inputs,outputs,real_state,target=target,optimizer=optimizer,learning_rate=learning_rate,steps=steps,jobname=jobname)
    build_input = compose(trained_parms,inputs)
    build_output = compose(trained_parms,outputs)
    return build_input,build_output

def infer_process_statistical(process:list,real_state,method,base_num,input_qubits=None,output_qubits=None,target='input',optimizer='quadratic',learning_rate=0.01,steps=5000,jobname='quadratic',device='simulate'):
    """ 通过量子状态向量构建量子态
    Args:
        process: 量子线路
        real_state: 真实的量子态
        method: 采样方法
        base_num: 采样基数
        input_qubits: 输入的量子比特
        target: 优化的目标，'input'或者'output'
        optimizer: 优化器，'quadratic'或者'descent'
        learning_rate: 学习率
        steps: 迭代次数
    Returns:
        build_input: 构建的输入量子态
        build_output: 构建的输出量子态
    """
    if input_qubits is None:
        input_qubits = ExcuteEngine.get_qubits(process)
    inputs,outputs = process_stat_sample(process,input_qubits,base_num=base_num,method=method,output_qubits=output_qubits,device=device)
    inputs = np.array(list(map(convert_state_to_density, inputs)))
    trained_parms = decompose(inputs,outputs,real_state,target=target,optimizer=optimizer,learning_rate=learning_rate,steps=steps,jobname=jobname)
    build_input = compose(trained_parms,inputs)
    build_output = compose(trained_parms,outputs)
    return build_input,build_output
    

def unit_test():
    """ 单元测试函数
    """
    from scipy.stats import unitary_group
    all_qubits = list(range(5))
    N_qubit = len(all_qubits)
    U = unitary_group.rvs(2**N_qubit)
    V = unitary_group.rvs(2**N_qubit)
    layer_circuit = [
        [{'name':'unitary','params': U,'qubits':all_qubits}],
        [{'name':'unitary','params': V,'qubits':all_qubits}],
    ]
    input_statevector = ExcuteEngine(layer_circuit[:1]).get_statevector()
    real_distribution = ExcuteEngine(layer_circuit).get_statevector()
    build_input,build_output = infer_process_statistical(layer_circuit[1:],input_statevector,optimizer='descent',method='random',base_num=2**3,target='input')
    print('real_output: ',real_distribution)
    print('build_output: ',build_output)
    print('real_input: ',input_statevector)
    print('build_input: ',build_input)
