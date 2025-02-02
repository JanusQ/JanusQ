from typing import Iterable
import copy
import pennylane as qml
import numpy as np
from .tomography.optimizers import SPSA,Adam
from .circuit_converter import layer_circuit_to_qml_circuit
from .tomography.variational_layer import Variational_circuit_MPS, Variational_state,RealAmplitudes
from .metric import fidelity
from tqdm import tqdm
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit import Aer
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeHanoi

def convert_state_to_density(state_vector):
    state_vector = state_vector.reshape(-1,1)
    density_matrix = state_vector @ state_vector.conj().T
    return density_matrix/density_matrix.trace()

def get_idling_backend():
    try:
        idle_backend = sorted(QiskitRuntimeService(channel="ibm_quantum").backends(simulator=False, operational=True, status_msg="active"), key=lambda x: x.status().pending_jobs)
        return idle_backend[0]
    except:
        return FakeHanoi()
def get_device(n_qubits,dev_type='an'):
    idle_backend = get_idling_backend()
    
    if dev_type == 'st':
        dev = qml.device("default.qubit", wires=n_qubits, shots=1000)
    elif dev_type == 'an':
        dev = qml.device("default.qubit", wires=n_qubits, shots=None)
    elif dev_type == 'ibmq':
        dev = qml.device('qiskit.ibmq', wires=n_qubits, backend=idle_backend.name,shots=10000)
    elif dev_type == 'ibmnoiseqasm':
        noise_model = NoiseModel.from_backend(idle_backend)
        dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
    return dev

def estimate_input(process: list,n_qubits,input_qubits: Iterable,output_target,output_qubits=None,max_iterations=50,dev_type='st',preparation_type='state'):
    '''
        通过对process 进行多次初始化，得到 多个 output,其线性叠加对应与整个process 的函数
        process: 电路的中间表示
        out_qubits: 输出的qubits
        input_qubits: 输入的qubits
        method: 采样的方法
        base_num: 采样的次数
        initial_label: 采样的初始态
        return:
            initialState: statevector
            new_circuit: initialized circuits
    '''
    n_qubits = len(input_qubits)
    dev = get_device(n_qubits,dev_type)
    if preparation_type == 'mps':
        params = 1.57*np.ones(3*(n_qubits-1)-2)
        state_preparation = Variational_circuit_MPS
    elif preparation_type == 'state':
        params = 1.57*np.ones(2**(n_qubits+1)-2)
        state_preparation = Variational_state
    elif preparation_type == 'real':
        reps= n_qubits  
        params = 1.57*np.ones((reps+1)*n_qubits)
        state_preparation = RealAmplitudes

    @qml.qnode(dev)
    def variational_prepare(params):
        state_preparation(params, input_qubits)
        layer_circuit_to_qml_circuit(process)
        return qml.density_matrix(output_qubits)
    infids= []

    fun   = lambda params : 1 - fidelity( variational_prepare(params),output_target)
    opt = SPSA()
    n_epsilons = 0
    with tqdm(range(max_iterations)) as pbar:
        for n in pbar:
            params = opt.step( fun, params )
            infids.append( fun(params) )
            pbar.set_description(f'SPSA optimizing, {n} round')
                    #设置进度条右边显示的信息
            pbar.set_postfix(fidelity = 1-infids[-1])
            n_epsilons += 1
            if infids[-1] < 5e-2:
                break
    
    n_epsilons = n_epsilons/(1-infids[-1])
    
    @qml.qnode(dev)
    def get_input_state(params):
        state_preparation(params, input_qubits)
        return qml.state()
    
    return get_input_state(params), infids[-1],n_epsilons


def estimate_output(process: list,n_qubits,output_qubits: Iterable,max_iterations=50,dev_type='an',preparation_type='state'):
    '''
        通过对process 进行多次初始化，得到 多个 output,其线性叠加对应与整个process 的函数
        process: 电路的中间表示
        out_qubits: 输出的qubits
        input_qubits: 输入的qubits
        method: 采样的方法
        base_num: 采样的次数
        initial_label: 采样的初始态
        return:
            initialState: statevector
            new_circuit: initialized circuits
    '''
    dev = get_device(n_qubits,dev_type)
    n_qubits = len(output_qubits)
    if preparation_type == 'mps':
        params = 1.57*np.ones(3*(n_qubits-1)-2)
        state_preparation = Variational_circuit_MPS
    elif preparation_type == 'state':
        params = 1.57*np.ones(2**(n_qubits+1)-2)
        state_preparation = Variational_state

    @qml.qnode(dev)
    def variational_prepare(params):
        layer_circuit_to_qml_circuit(process)
        state_preparation(params, output_qubits)
        return qml.probs(wires=output_qubits)
    infids= []
    fun   = lambda params : 1 - variational_prepare(params)[0]
    
    opt = SPSA()
    with tqdm(range(max_iterations)) as pbar:
        for n in pbar:
            params = opt.step( fun, params )
            infids.append( fun(params) )
            pbar.set_description(f'SPSA optimizing, {n} round')
                    #设置进度条右边显示的信息
            pbar.set_postfix(fidelity = 1- infids[-1])
            if infids[-1] < 1e-5:
                break
    n = n/(1-infids[-1])

    @qml.qnode(dev)
    def get_output_state(params):
        state_preparation(params, output_qubits)
        return qml.state()
    
    return get_output_state(params), infids[-1],n

