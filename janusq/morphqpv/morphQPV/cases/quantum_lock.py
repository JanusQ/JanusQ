
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector
from morphQPV.execute_engine.excute import ExcuteEngine, convert_state_to_density
from morphQPV.execute_engine.metric import fidelity
from morphQPV.assume_guarantee.inference import compose
import copy
from qiskit.quantum_info import Statevector
from functools import reduce
from itertools import product
from typing import Iterable
from tqdm import tqdm
from functools import partial
from morphQPV.assume_guarantee.optimizer.linear_solver_lock import LinearSolver
from morphQPV.assume_guarantee.optimizer.sgd import SgdOptimizer


def decompose(inputs, outputs, real_target, target='input', optimizer='quadratic', learning_rate=0.01, steps=10000):
    """ 优化参数
    Args:
        inputs: 输入的量子态
        outputs: 输出的量子态
        real_target: 真实的目标态
        target: 优化的目标，'input'或者'output'
        optimizer: 优化器，'quadratic'或者'descent'
        learning_rate: 学习率
        steps: 迭代次数
    Returns:
        best_params: 最优参数
    """
    if optimizer == 'quadratic':
        model = LinearSolver()
        model.Params.LogToConsole = 1
        model.Params.TimeLimit = 300
        best_params = model.solve(inputs, outputs, real_target, target)
        return best_params
    elif optimizer == 'descent':
        model = SgdOptimizer(step_size=learning_rate, steps=steps)
        best_params = model.optimize_params(
            inputs, outputs, real_target, target)
        return best_params



def get_inititial_statevectors(n_qubits: int, know_state=None, num=10):
    dim = 2**n_qubits
    # using Gram-Schmidt  orthogonalization
    if know_state is None:
        know_state = np.random.rand(dim)
        know_state = know_state/np.linalg.norm(know_state, ord=2)
    us = [know_state]
    for i in range(num):
        state = np.random.rand(dim)
        state = state/np.linalg.norm(state, ord=2)
        for u in us:
            state = state - np.dot(state.conj().T, u)*u
        state = [abs(s) if abs(s) > 1e-5 else 0 for s in state]
        state = state/np.linalg.norm(state, ord=2)
        us.append(state)
    us = [u.astype(np.complex128) for u in us]
    print(np.linalg.matrix_rank(np.array(us[1:])))
    return us[1:]

def gen_statevector_from_subspace(n_qubits,subspaces=None,num=10):
    dim = 2**n_qubits
    state = np.random.randn(dim)
    state = state/np.linalg.norm(state, ord=2)
    if subspaces is not None:
        for subspace in subspaces:
            state = state - np.dot(state.conj().T, subspace)*subspace
        state = state/np.linalg.norm(state, ord=2)
    return state.astype(np.complex128)





def convert_lock_density_to_state(density_matrix):
    diags = np.diag(density_matrix)
    return np.sqrt(diags)


def convert_lock_state_to_density(state):
    return np.diag(state**2)


def process_sample(process: list, input_qubits: Iterable, method: str = 'base', base_num: int = 10, initial_state=None):
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
    if method == 'base':
        n_cumulate = 0
        for state in product(['0', '1'], repeat=n_qubits):
            new_circ = copy.deepcopy(process)
            state = reduce(lambda x, y: x+y, state)
            statevec = Statevector.from_label(state).data
            new_circ.insert(
                0, [{'name': 'initialize', 'params': statevec, 'qubits': input_qubits}])
            n_cumulate += 1
            if np.allclose(statevec, initial_state):
                continue
            if n_cumulate > base_num:
                break
            yield statevec, new_circ

    elif method == 'random':
        initial_states = get_inititial_statevectors(
            n_qubits, know_state=initial_state, num=base_num)
        for input_state in tqdm(initial_states, desc='producing initialied circuits'):
            new_circ = copy.deepcopy(process)
            new_circ.insert(
                0, [{'name': 'initialize', 'params': input_state, 'qubits': input_qubits}])
            yield input_state, new_circ

    else:
        raise NotImplementedError("method is not implemented")


def infer_process_by_statevector(process: list, real_state, method, base_num, input_qubits=None, output_qubits=None, target='input', optimizer='quadratic', learning_rate=0.01, steps=5000, know_key=None, hidden_key=None):
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

    inputs = []
    if input_qubits is None:
        input_qubits = ExcuteEngine.get_qubits(process)
    circs = []
    for i, circ in process_sample(process, input_qubits, base_num=base_num, method=method, initial_state=Statevector.from_label(know_key).data):
        inputs.append(i)
        circs.append(circ)
    if output_qubits is None:
        outputs = list(
            map(partial(ExcuteEngine.excute_on_pennylane, type='statevector'), circs))
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        trained_parms = decompose(inputs, outputs, real_state, target=target,
                                  optimizer=optimizer, learning_rate=learning_rate, steps=steps)
        build_input = compose(trained_parms, inputs)
        build_output = compose(trained_parms, outputs)
        return build_input, build_output
    else:
        outputs = list(map(partial(ExcuteEngine.excute_on_pennylane,
                       type='density', output_qubits=output_qubits), circs))
        outputs = list(map(convert_lock_density_to_state, outputs))
        for i in range(len(inputs)):
            assert np.allclose(inputs[i][hidden_key], outputs[i][1])
        trained_parms = decompose(inputs, outputs, real_state, target=target,
                                  optimizer=optimizer, learning_rate=learning_rate, steps=steps)
        build_input = compose(trained_parms, inputs)
        build_output = compose(trained_parms, outputs)
        return build_input, build_output


def flip_sign(wires, arr_bin: str):
    # generate the unitary matrix
    n_qubits = len(wires)
    assert len(arr_bin) == n_qubits
    flip_sign_matrix = np.eye(2**n_qubits)
    idx = int(''.join(arr_bin), 2)
    flip_sign_matrix[idx, idx] = -1
    return flip_sign_matrix


def convert_density_to_state(density_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    idx = np.argmax(eigenvalues)
    return eigenvectors[:, idx]

def generate_qml_lock_circuit(states_qubits_num, key_state, hidden_key=None, input_state=None):
    if input_state is None:
        layer_circuit = [[{'name': 'h', 'qubits': [0]}]]
    else:
        layer_circuit = [[{'name': 'initialize', 'qubits': list(range(
            1, states_qubits_num+1)), 'params': input_state}, {'name': 'h', 'qubits': [0]}]]
    assert len(key_state) == states_qubits_num
    layer_circuit.append([{'name': 'flipkey', 'qubits': list(range(states_qubits_num+1)), 'ctrl_qubits': [0],
                         'ctrled_qubits':list(range(1, states_qubits_num+1)), 'params' : key_state}])
    if hidden_key is not None:
        assert len(hidden_key) == states_qubits_num
        layer_circuit.append([{'name': 'flipkey', 'qubits': list(range(states_qubits_num + 1)), 'ctrl_qubits': [0], 'ctrled_qubits':list(
            range(1, states_qubits_num+1)), 'params': hidden_key}])
    layer_circuit.append([{'name': 'h', 'qubits': [0]}])
    return layer_circuit

def generate_lock_circuit(states_qubits_num, key_state, hidden_key=None, input_state=None):
    if input_state is None:
        layer_circuit = [[{'name': 'h', 'qubits': [0]}]]
    else:
        layer_circuit = [[{'name': 'initialize', 'qubits': list(range(
            1, states_qubits_num+1)), 'params': input_state}, {'name': 'h', 'qubits': [0]}]]
    assert len(key_state) == states_qubits_num
    layer_circuit.append([{'name': 'ctrl', 'qubits': list(range(states_qubits_num+1)), 'ctrl_qubits': [0],
                         'ctrled_qubits':list(range(1, states_qubits_num+1)), 'params':flip_sign(list(range(states_qubits_num)), key_state)}])
    if hidden_key is not None:
        assert len(hidden_key) == states_qubits_num
        layer_circuit.append([{'name': 'ctrl', 'qubits': list(range(states_qubits_num+1)), 'ctrl_qubits': [0], 'ctrled_qubits':list(
            range(1, states_qubits_num+1)), 'params':flip_sign(list(range(states_qubits_num)), hidden_key)}])
    layer_circuit.append([{'name': 'h', 'qubits': [0]}])
    return layer_circuit

def quantumlock_simulate(lock,hidden,inputstate):
    probs = inputstate[int(lock,2)]**2 + inputstate[int(hidden,2)]**2
    return probs

def state_projected(state, determined_state):
    """ project state to the subspace without determined_state """
    coeff = np.dot(state.conj().T, determined_state)
    eventual_state = state - coeff*determined_state
    if np.linalg.norm(eventual_state, ord=2) > 1e-5:
        eventual_state = eventual_state/np.linalg.norm(eventual_state, ord=2)
        return eventual_state
    else:
        return None

def get_input_state(states):
    # superposition of all states
    state = np.zeros(2**len(states[0]))
    for s in states:
        state[int(s, 2)] = 1
    state = state/np.linalg.norm(state, ord=2)
    return state
def get_bit_string(n_qubit):
    return ''.join([str(np.random.randint(2)) for _ in range(n_qubit)])

def infer_circuit(stateset,n_qubit,sampletimes,lock,hidden,pbar):
    if len(stateset) == 1:
        return stateset[0],sampletimes
    left = stateset[:len(stateset)//2]
    right = stateset[len(stateset)//2:]
    leftinput = get_input_state(left)
    infered_probs = quantumlock_simulate(lock,hidden,leftinput)
    sampletimes += 1
    pbar.update(len(left))
    if np.sum(infered_probs) == 0:
        return infer_circuit(right, n_qubit, sampletimes,lock,hidden,pbar)
    else:
        return infer_circuit(left, n_qubit, sampletimes,lock,hidden,pbar)

def quantum_backdoor_attack(inference_circuit, lock, hidden, n_qubit, base_num=2**4, method='random', opt='quadratic'):
    ## using divice and conquer to find the hidden key
    rightoutput = Statevector.from_label('1').data
    ## generate all bit strings
    stateset = [''.join(bit) for bit in product(['0', '1'], repeat=n_qubit)]
    ## move lock key from the state set
    stateset.remove(lock)
    # divede the state set into two parts
    with tqdm(total=len(stateset), desc='infering state vector in quantum lock') as pbar:
        infer_hidden_key,samples = infer_circuit(stateset, n_qubit,6,lock, hidden,pbar)
    assert infer_hidden_key == hidden
    return 2**((samples)/2)


def quantum_backdoor_discovering(inference_circuit, lock, hidden, n_qubit, base_num=2**4, method='random', opt='quadratic'):
    output = Statevector.from_label('1').data
    infered_input, infered_output = infer_process_by_statevector(inference_circuit, output, input_qubits=list(range(
        1, n_qubit+1)), output_qubits=[0], method=method, base_num=base_num, target='output', know_key=lock, optimizer=opt, hidden_key=int(hidden, 2))
    input_state_found = state_projected(
        infered_input, Statevector.from_label(lock).data)
    if input_state_found is None:
        return False
    else:
        lock_circuit = generate_lock_circuit(
            n_qubit, lock, input_state=input_state_found, hidden_key=hidden)
        output_density = ExcuteEngine(
            lock_circuit).get_density_matrix(qubits=[0])
        # assert fidelity(
        #     output_density, convert_lock_state_to_density(infered_output)) > 0.9
        one_density = convert_state_to_density(output)
        assert np.allclose(fidelity(output_density, one_density),
                           output_density[1][1])
        if fidelity(output_density, one_density) > 0.5:
            return True
        else:
            return False
