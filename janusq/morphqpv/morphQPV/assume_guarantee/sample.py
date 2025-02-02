import copy
from qiskit.quantum_info import random_clifford,Statevector
from functools import reduce
from itertools import product
from typing import Iterable
from tqdm import tqdm
from functools import partial
from morphQPV.execute_engine.excute import ExcuteEngine,qiskit_circuit_to_layer_cirucit,convert_density_to_state
from qiskit import QuantumCircuit, Aer, execute
import numpy as np
# import jax
# import jax.numpy as jnp

# @jax.jit
def get_inititial_circuit(n_qubits:int):
    clifford = random_clifford(n_qubits)
    return clifford.to_circuit()
def get_inititial_statevector(circuit):
    simulator = Aer.get_backend('statevector_simulator')
    job = execute(circuit, simulator)
    statevector = job.result().get_statevector()
    return statevector.data

def get_circuit_from_label(label:str,n_qubits:int):
    layer_circuit = [[],[]]
    for i in range(n_qubits):
        if label[i] == '0':
            layer_circuit[0].append({'name':'x','qubits':[i]})
        elif label[i] == '+':
            layer_circuit[0].append({'name':'h','qubits':[i]})
        elif label[i] == 'r':
            layer_circuit[0].append({'name':'h','qubits':[i]})
            layer_circuit[1].append({'name':'s','qubits':[i]})
    return layer_circuit

def build_relation(process,input_qubits,output_qubits,**config):
    sampleconfig = {}
    sampleconfig['method'] = config['sample_method']
    sampleconfig['base_num'] =min(config['base_num'],2**(len(input_qubits)+1))
    sampleconfig['device'] = config['device']
    input_states,output_states = process_sample(process,input_qubits,output_qubits= output_qubits,**sampleconfig)
    output_states = list(map(convert_density_to_state, output_states))
    return input_states,output_states

def process_sample(process: list,input_qubits: Iterable,output_qubits=None,method:str='base',base_num:int=10,clifford_tomography=False,device= 'simulate'):
    '''
        通过对process 进行多次初始化，得到 多个 output,其线性叠加对应与整个process 的函数
        process: 电路的中间表示
        out_qubits: 输出的qubits
        input_qubits: 输入的qubits
        method: 采样的方法
        base_num: 采样的次数
        initial_label: 采样的初始态
        device: 采样的设备
            ibmq: ibm 的 真实量子计算机
            simulate: 本地模拟器
        return:
            initialState: statevector
            new_circuit: initialized circuits
    '''
    n_qubits = len(input_qubits)
    if clifford_tomography:
        get_output = lambda  state: ExcuteEngine.excute_on_pennylane(process,type='definedinput',shots=1000,output_qubits=output_qubits,input_qubits=input_qubits,input_state=state)
    get_output = lambda  state_circuit: ExcuteEngine.output_state_tomography(state_circuit+process,output_qubits=output_qubits,device=device)
    if method == 'base':
        labels = [''.join(state) for state in product(['0','1','+','r'],repeat=n_qubits)]
        labels = labels[:base_num]
        input_states =  [Statevector.from_label(state).data for state in  labels]
        input_circuits = [get_circuit_from_label(label,n_qubits) for label in labels]
    if method == 'basis':
        labels = [''.join(state) for state in product(['0','1'],repeat=n_qubits)][:base_num]
        input_states =  [Statevector.from_label(state).data for state in  labels]
        input_circuits = [get_circuit_from_label(label,n_qubits) for label in labels]
    elif method == 'random':
        input_circuits = list(map(get_inititial_circuit,tqdm([n_qubits]*base_num,desc='producing input circuits for random sampling')))
        input_states = list(map(get_inititial_statevector,tqdm(input_circuits,desc='producing input states for random sampling')))
        input_circuits = list(map(lambda x: qiskit_circuit_to_layer_cirucit(x),input_circuits))
    
    output_states = list(map(get_output,tqdm(input_circuits,desc='producing output states')))
    return input_states , output_states

   
def process_stat_sample(process: list,input_qubits: Iterable,method:str='base',base_num:int=10,output_qubits=None,clifford_tomography=False,device= 'ibmq'):
    '''
        通过对process 进行多次初始化，得到 多个 output,其线性叠加对应与整个process 的函数
        process: 电路的中间表示
        out_qubits: 输出的qubits
        input_qubits: 输入的qubits
        method: 采样的方法
        base_num: 采样的次数
        initial_label: 采样的初始态
        device: 采样的设备
            ibmq: ibm 的 真实量子计算机
            simulate: 本地模拟器
        return:
            initialState: statevector
            new_circuit: initialized circuits
    '''
    n_qubits = len(input_qubits)
    get_output = lambda  state_circuit: ExcuteEngine.output_state_statistic(state_circuit+process,input_qubits,device=device)
    if method == 'base':
        labels = [''.join(state) for state in product(['0','1','+','r'],repeat=n_qubits)][:base_num]
        input_states =  [Statevector.from_label(state).data for state in  labels]
        input_circuits = [get_circuit_from_label(label,n_qubits) for label in labels]
    if method == 'basis':
        labels = [''.join(state) for state in product(['0','1'],repeat=n_qubits)][:base_num]
        input_states =  [Statevector.from_label(state).data for state in  labels]
        input_circuits = [get_circuit_from_label(label,n_qubits) for label in labels]
    elif method == 'random':
        input_circuits = list(map(get_inititial_circuit,tqdm([n_qubits]*base_num,desc='producing input circuits for random sampling')))
        input_states = list(map(get_inititial_statevector,tqdm(input_circuits,desc='producing input states for random sampling')))
        input_circuits = list(map(lambda x: qiskit_circuit_to_layer_cirucit(x),input_circuits))
    
    output_states = list(map(get_output,tqdm(input_circuits,desc='producing output states')))
    return input_states , output_states




