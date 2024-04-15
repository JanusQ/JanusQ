from collections import defaultdict
import copy
import itertools
import math
import time
import traceback
from random import choice as random_choice, sample
from random import randint

import cloudpickle as pickle
import jax
import numpy as np
import optax
import pennylane as qml

from jax import numpy as jnp
from jax import vmap
from jax.config import config
from qiskit import QuantumCircuit, transpile
from sklearn.neighbors import NearestNeighbors

import ray

import logging

from tqdm import tqdm

from data_objects.backend import Backend, FullyConnectedBackend
from data_objects.circuit import Circuit, Layer, circuit_to_qiskit, qiskit_to_circuit
from tools.optimizer import OptimizingHistory
from tools.ray_func import wait

# TODO: gates太多了jit的compile会很慢
# TODO: 每个node都要分开来算才能快

config.update("jax_enable_x64", True)

'''load data for decompostion'''
IPARAMS = {}
with open('./analysis/decomposition_data/identity_params.pkl', 'rb') as file:
    IPARAMS: dict = pickle.load(file)


RFS = {}  # redundancy-free candidate set
for n_qubits in range(4, 7):
    with open(f'./analysis/decomposition_data/{n_qubits}_crz_best.pkl', 'rb') as file:
        circuits = pickle.load(file)

    for circuit in circuits:
        for layer in circuit:
            for gate in layer:
                gate['params'] = np.zeros(len(gate['params']))
                
    RFS[n_qubits] = [
        Circuit(circuit, n_qubits)
        for circuit in circuits
    ]


def reshape_unitary_params(params: np.ndarray, n_qubits: int) -> np.ndarray:
    n_params = 4**n_qubits
    return (params[:n_params] + params[n_params:] * 1j).reshape((2**n_qubits, 2**n_qubits))


@jax.jit
def params_to_unitary(params: jnp.ndarray) -> jnp.ndarray:
    z = 1/jnp.sqrt(2)*params
    q, r = jnp.linalg.qr(z)  # numpy的qr没办法求导
    d = r.diagonal()
    q *= d/jnp.abs(d)
    return q


@jax.jit
def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    return jnp.abs(1 - jnp.abs(jnp.sum(jnp.multiply(A, jnp.conj(B)))) / A.shape[0])


def create_unitary_gate(connect_qubits):
    if len(connect_qubits) == 1:
        return [[{'name': 'u', 'qubits': list(connect_qubits), 'params': np.zeros(3), }]]
    else:
        # 现在的收敛方式似乎是有问题的
        n_connect_qubits = len(connect_qubits)
        return [[{
            'name': 'unitary',
            'qubits': list(connect_qubits),  # 不知道有没有参数量更少的方法
            'params': np.array(IPARAMS.get(n_connect_qubits, np.random.rand((4 ** n_connect_qubits) * 2))),
        }]]


def circuit_to_pennylane(circuit: Circuit, params=None, offest=0):
    point = 0
    for layer in circuit:
        for gate in layer:
            qubits = [q+offest for q in gate['qubits']]
            if gate['name'] == 'u':
                if params is None:
                    theta, phi, lam = gate['params']
                else:
                    theta, phi, lam = params[point: point+3]
                    point += 3
                qml.U3(theta, phi, lam, wires=qubits)
            elif gate['name'] == 'cx':
                qml.CNOT(wires=qubits)
            elif gate['name'] == 'cz':
                qml.CZ(wires=qubits)
            elif gate['name'] == 'crz':
                if params is None:
                    phi = gate['params'][0]
                else:
                    phi = params[point]
                    point += 1
                qml.CRZ(phi, wires=qubits)
            elif gate['name'] == 'unitary':
                n_qubits = len(qubits)
                n_params = (4**n_qubits)*2
                if params is None:
                    unitary_params = gate['params']
                else:
                    unitary_params = params[point: point+n_params]
                point += n_params
                unitary_params = reshape_unitary_params(
                    unitary_params, n_qubits)
                unitary = params_to_unitary(unitary_params)
                qml.QubitUnitary(unitary, wires=qubits)
            else:
                logging.error('Unkown gate type', gate)
                # raise Exception('Unkown gate type', gate)


'''TODO: 会出现比特没有被用到然后矩阵算错的情况'''


def circuit_to_matrix(circuit: Circuit, n_qubits, params=None) -> jax.numpy.array:
    if len(circuit) == 0:
        return jnp.eye(2**n_qubits)
    with qml.tape.QuantumTape() as U:
        circuit_to_pennylane(
            circuit, params=params, offest=0)
    return qml.matrix(U, wire_order=list(range(n_qubits)))


def assign_params(params, circuit: Circuit) -> Circuit:
    circuit = circuit.copy()
    count = 0
    for gates in circuit:
        for gate in gates:
            for index, _ in enumerate(gate['params']):
                gate['params'] = np.array(gate['params'])
                gate['params'][index] = params[count]
                count += 1
    return circuit


def find_parmas(n_qubits: int, circuit: Circuit, U: np.ndarray, lr=1e-1, max_epoch=1000, allowed_dist=1e-2, n_iter_unchange=10,
                unchange_tol=1e-2, verbose=False, reset_params=False) -> tuple[Circuit, float]:
    
    # verbose = True
    np.random.RandomState()
    
    assert len(circuit) != 0

    params = []
    if reset_params:
        for layer in circuit:
            for gate in layer:
                params += list(create_unitary_gate(gate['qubits'])[0][0]['params'])
    else:
        for layer in circuit:
            for gate in layer:
                params += list(gate['params'])
    params = jnp.array(params)

    @jax.jit
    def cost_hst(params, target_U):
        return matrix_distance_squared(circuit_to_matrix(circuit, n_qubits, params), target_U)

    best_params = params
    min_loss = cost_hst(params, U)
    for _ in range(3):
        opt_history = OptimizingHistory(
            params, lr, unchange_tol, n_iter_unchange, max_epoch, allowed_dist, verbose)

        opt = optax.adamw(learning_rate=lr)
        opt_state = opt.init(params)

        while True:
            loss_value, gradient = jax.value_and_grad(cost_hst)(params, U)

            opt_history.update(loss_value, params)

            updates, opt_state = opt.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)

            if opt_history.should_break:
                break

        lr = opt_history.min_loss/10
        params = opt_history.best_params

        max_epoch = max_epoch // 2

        if min_loss > opt_history.min_loss:
            min_loss = opt_history.min_loss
            best_params = opt_history.best_params

    circuit = assign_params(best_params, circuit)
    return Circuit(circuit, n_qubits), min_loss


def optimize(now_circuit: Circuit, new_layers: list[Layer], n_optimized_layers, U, lr, n_iter_unchange, n_qubits, allowed_dist, old_dist,
             remote=False, verbose=False, reset_params=False) -> tuple[jnp.ndarray, Circuit, float]:

    if not remote:
        now_circuit = copy.deepcopy(now_circuit)
        new_layers = copy.deepcopy(new_layers)

    total_circuit = now_circuit + new_layers
    unoptimized_layers, optimized_layers = total_circuit[:-n_optimized_layers], total_circuit[-n_optimized_layers:]
    unoptimized_U = circuit_to_matrix(unoptimized_layers, n_qubits)

    optimized_layers, best_dist = find_parmas(n_qubits, optimized_layers, U @ unoptimized_U.T.conj(), max_epoch=500,
                                         allowed_dist=allowed_dist, n_iter_unchange=n_iter_unchange,
                                         unchange_tol=old_dist / 1000, lr=lr,
                                         verbose=verbose, reset_params=reset_params)

    total_circuit = unoptimized_layers + optimized_layers

    remained_U = None

    return remained_U, Circuit(total_circuit, now_circuit.n_qubits), best_dist


class ApproaxitationNode():
    def __init__(self, target_U: np.array, former_circuit: Circuit, inserted_layers: list[Layer], iter_count: int, config: dict, former_node=None):
        former_circuit = copy.deepcopy(former_circuit)
        inserted_layers = copy.deepcopy(inserted_layers)

        self.n_qubits = int(math.log2(target_U.shape[0]))
        self.target_U = target_U

        self.former_circuit = former_circuit
        self.inserted_layers = inserted_layers
        self.circuit = former_circuit + inserted_layers

        assert self.circuit.n_qubits == 4
        
        config = dict(config)
        self.config = config
        self.former_node: ApproaxitationNode = former_node
        self.son_nodes: list[ApproaxitationNode] = []
        self.iter_count = iter_count

        self.backend: Backend = config['backend']
        self.n_qubits = self.backend.n_qubits
        self.allowed_dist = config['allowed_dist']
        self.logger = config['logger']
        self.n_candidates = config['n_candidates']

        self.distance: float = None
        self.n_gates: int = None

        self.before_optimize_dist = 1
        self.after_optimize_dist = None
        if former_node is not None:
            self.before_optimize_dist: float = former_node.after_optimize_dist

    # 记录下是否被optimze了
    def optimize(self, n_optimized_layers=None, n_iter_unchange=20, multi_process=None, verbose=False):
        if multi_process is None:
            multi_process = self.config['multi_process']

        if n_optimized_layers is None:
            if self.before_optimize_dist < 1e-1:
                n_iter_unchange = 20  # lr = .01
                n_optimized_layers = len(
                    self.former_circuit) + len(self.inserted_layers)
            else:
                n_iter_unchange = 10  # lr = .1
                n_optimized_layers = 20 + len(self.inserted_layers)

        lr = max([self.before_optimize_dist / 5, 1e-2])  # TODO: 需要尝试的超参数

        reset_params = False
        if self.iter_count % 10 == 0:
            reset_params = True
        if self.former_node is not None and self.former_node.before_optimize_dist > 0.1 and self.former_node.after_optimize_dist < 0.1:
            reset_params = True

        if multi_process:
            return optimize_remote.remote(self.former_circuit, self.inserted_layers, n_optimized_layers, self.target_U, lr,
                                          n_iter_unchange, self.n_qubits, self.allowed_dist, self.before_optimize_dist, verbose=verbose, reset_params=reset_params)
        else:
            return optimize(self.former_circuit, self.inserted_layers, n_optimized_layers, self.target_U, lr, n_iter_unchange,
                            self.n_qubits, self.allowed_dist, self.before_optimize_dist, verbose=verbose, reset_params=reset_params)

    def update(self, remained_U, circuit, dist):
        '''更新下reward'''
        self.remained_U = remained_U
        self.circuit = circuit
        self.after_optimize_dist = dist

        penalty = 1
        for index, layer in enumerate(self.inserted_layers):
            for gate in layer:
                penalty += 1

        dist_decrement = self.before_optimize_dist - self.after_optimize_dist

        if dist_decrement > 0:
            dist_decrement /= penalty
        else:
            dist_decrement *= penalty

        self.reward = dist_decrement

    @property
    def estimated_n_gates(self):
        n_gates = 0
        for layer in self.circuit:
            for gate in layer:
                n_gates += 1
        return n_gates

    def expand(self):
        config = self.config

        # self.logger.warning(f'Expanding iter_count = {self.iter_count}')

        backend: Backend = config['backend']
        n_candidates: int = config['n_candidates']

        canditate_layers: list[list[Layer]] = []
        if self.iter_count != 0:
            canditate_layers = [[]]  # 加一个空的

        '假设只有u, crz'
        canditate_layers += RFS[self.n_qubits]

        '''去掉重复的'''
        canditate_layers = remove_repeated_candiate(canditate_layers)

        futures = []
        for candidate_layer in canditate_layers:
            son_node = ApproaxitationNode(self.target_U, self.circuit,
                                          candidate_layer, self.iter_count+1, self.config, self)
            self.son_nodes.append(son_node)

            futures.append((son_node, son_node.optimize()))

        for son_node, future in wait(futures):
            son_node.update(*future)

        # TODO: 需要确定一下至少一个是大于0的
        rewards = np.array([son_node.reward for son_node in self.son_nodes])
        if np.all(rewards < 0):
            self.logger.error(f'No improvement after inserting gates')

        # 增加一些随机性，靠前的比较好的都考虑进去
        best_son_node: ApproaxitationNode = self.son_nodes[np.argmax(rewards)]

        self.logger.warning(
            f'iter_count= {self.iter_count} now_dist= {best_son_node.after_optimize_dist}, #gate = {best_son_node.estimated_n_gates} \n')

        return best_son_node

    def get_circuit(self) -> Circuit:
        for layer_gates in self.circuit:
            for gate in layer_gates:
                assert gate['name'] != 'unitary'

        qiskit_circuit: QuantumCircuit = circuit_to_qiskit(self.circuit, barrier=False)

        qiskit_circuit = transpile(qiskit_circuit, optimization_level=3, basis_gates=self.backend.basis_gates, initial_layout=[
                                   qubit for qubit in range(self.n_qubits)])

        return qiskit_to_circuit(qiskit_circuit)


def remove_repeated_candiate(canditates: list[list[Layer]]):
    new_canditate_layers = []
    for candidate in canditates:
        former_tuple = []
        next_former_tuple = []
        for index, layer in enumerate(candidate):
            new_layer = []
            for gate in layer:
                qubits = tuple(gate['qubits'])
                if qubits not in former_tuple:
                    new_layer.append(gate)
                next_former_tuple.append(qubits)
            former_tuple = next_former_tuple
            next_former_tuple = []
            candidate[index] = new_layer
        candidate = [layer for layer in candidate if len(layer) != 0]
        new_canditate_layers.append(candidate)
    canditates = new_canditate_layers

    def hash_circuit(circuit):
        circuit_tuple = []

        for layer in circuit:
            layer_tuple = []
            for gate in layer:
                layer_tuple.append(tuple(gate['qubits']))
            layer_tuple.sort()
            circuit_tuple.append(tuple(layer_tuple))

        return tuple(circuit_tuple)

    hash2canditate = {
        hash_circuit(candidate_layer): candidate_layer
        for candidate_layer in canditates
    }

    return list(hash2canditate.values())


def decompose_to_small_unitaries(target_U, backend: Backend, allowed_dist=1e-5, multi_process=True, n_candidates=10, logger=logging.INFO):

    # n_block: 一个candicate拆解成的block数量

    n_qubits = int(math.log2(len(target_U)))

    if n_qubits < 7:
        block_size = 4  # number of qubits in each block
    else:
        block_size = 5

    n_qubits2n_blocks = {
        5: 4,
        6: 11,
        7: 17,
        8: 40,
    }
    now_n_blocks = n_qubits2n_blocks[n_qubits]

    solutions = []
    success_n_blocks = []
    fail_n_blocks = [0]

    while True:
        connected_qubit_sets = backend.get_connected_qubit_sets(block_size)
        candidates = []
        for _ in range(n_candidates):
            candidate = []
            for _ in range(now_n_blocks):  # 感觉4比特，block_size为3的时候不需要两层
                candidate_qubits = random_choice(connected_qubit_sets)
                candidate += create_unitary_gate(candidate_qubits)
            candidates.append(candidate)

        candidates: list[Circuit] = remove_repeated_candiate(
            candidates)  # 去掉重复的

        candidates = [
            candidate
            for candidate in candidates
            if len([gate for layer in candidate for gate in layer]) == now_n_blocks
        ]

        futures = []
        for candidate in candidates:
            kwargs = {
                'n_qubits': n_qubits,
                'circuit': candidate,
                'U': target_U,
                'max_epoch': 1500,
                'allowed_dist': 1e-5,
                'n_iter_unchange': 500,
                'unchange_tol': 1e-5,
                'lr': 0.1,
                'verbose': False,
            }
            if multi_process:
                futures.append(find_parmas_remote.remote(**kwargs))
            else:
                futures.append(find_parmas(**kwargs))

        solutions_now_n_blocks = []
        for candidate, min_loss in wait(futures):
            if min_loss <= allowed_dist:
                solutions_now_n_blocks.append((candidate, min_loss))
        solutions_now_n_blocks.sort(key=lambda elm: elm[1])

        if len(solutions_now_n_blocks) != 0:
            solutions = solutions_now_n_blocks
            success_n_blocks.append(now_n_blocks)
            now_n_blocks = (min(success_n_blocks) + max(fail_n_blocks))//2
        else:
            fail_n_blocks.append(now_n_blocks)
            if now_n_blocks >= n_qubits2n_blocks[n_qubits]:
                now_n_blocks += 1
                logging.warning(f'n_blocks of {n_qubits} qubits should be {n_qubits2n_blocks[n_qubits]}')
            else:
                now_n_blocks = (min(success_n_blocks) + max(fail_n_blocks))//2

        if now_n_blocks in (success_n_blocks + fail_n_blocks):  # 已经都搜索过了
            break

    if n_qubits == 4:
        return solutions

    # futher decompose for larger unitary
    solutions = solutions[:7]
    for candidate, min_loss in solutions:
        for layer in candidate:
            for gate in layer:
                gate_qubits = gate['qubits']
                if len(gate_qubits) <= 4:
                    continue

                unitary_params = gate['params']

                unitary_params = reshape_unitary_params(
                    unitary_params, len(gate_qubits))
                gate_unitary = params_to_unitary(unitary_params)

                sub_backend = FullyConnectedBackend(n_qubits=len(
                    gate_qubits), basis_single_gates=['u'], basis_two_gates=['crz'])

                kwargs = {
                    'target_U': gate_unitary,
                    'backend': sub_backend,
                    'allowed_dist': allowed_dist/2,
                    'multi_process': multi_process,
                    'n_candidates': 10,
                    'logger_level': logger,
                }
                if multi_process:
                    gate['result'] = decompose_to_small_unitaries_remote.remote(
                        **kwargs)
                else:
                    gate['result'] = decompose_to_small_unitaries(**kwargs)

    futures = []

    for candidate, min_loss in solutions:
        overall_circuit = []
        for layer in candidate:
            for gate in layer:
                gate_qubits = gate['qubits']
                if len(gate_qubits) <= 4:
                    overall_circuit += [[gate]]
                    continue

                gate_circuits = wait(gate['result'])
                synthesized_gate = gate_circuits[0]
                for _layer_gates in synthesized_gate:
                    for _gate in _layer_gates:
                        _gate['qubits'] = [gate_qubits[_qubit]
                                           for _qubit in _gate['qubits']]

                overall_circuit += synthesized_gate

        kwargs = {
            'n_qubits': n_qubits,
            'circuit': overall_circuit,
            'U': target_U,
            'max_epoch': 1500,
            'allowed_dist': 1e-10,
            'n_iter_unchange': 500,
            'unchange_tol': 1e-10,
            'lr': 0.1,
            'verbose': True
        }
        if multi_process:
            futures.append(find_parmas_remote.remote(**kwargs))
        else:
            futures.append(find_parmas(**kwargs))

    n_4q_to_solutions = defaultdict(list)
    for future in futures:
        circuit, min_loss = wait(future)
        n_4q = len([gate for layer in circuit for gate in layer])
        n_4q_to_solutions[n_4q].append(circuit)

    min_n4q = min(n_4q_to_solutions.keys())

    logging.info(f'Decompose to {min_n4q} 4-q unitaries. The number of solutions are {len(n_4q_to_solutions[min_n4q])}')
    return n_4q_to_solutions[min_n4q]


def decompose_to_gates(target_U, backend: Backend, allowed_dist=1e-5, multi_process=True, n_candidates=40, logger=None) -> list[Circuit]:
    assert backend.basis_two_gates == ['crz'], backend.basis_two_gates

    n_qubits = int(math.log2(len(target_U)))

    I = jnp.eye(2 ** n_qubits)
    I_dist = matrix_distance_squared(target_U, I)
    if I_dist <= allowed_dist:
        return Circuit([], n_qubits=n_qubits)

    root_node = ApproaxitationNode(target_U, Circuit([], n_qubits), [], 0, {
        'multi_process': multi_process,
        'allowed_dist': allowed_dist,
        'backend': backend,
        'synthesis_start_time': time.time(),
        'logger': logger,
        'logger_level': logger,
        'n_candidates': n_candidates,   # 逼近的候选数量，可能需要随比特数变化变化
    })
    root_node.after_optimize_dist = I_dist

    now_node = root_node

    max_solution_nodes = 1

    all_nodes: list[ApproaxitationNode] = []
    solution_nodes: list[ApproaxitationNode] = []  # 已经找到的分解
    now_node: ApproaxitationNode = root_node

    # [340, 326, 326, 354, 368] 差距还是有的，但是不是特别大
    while len(solution_nodes) < max_solution_nodes:
        while now_node.after_optimize_dist >= allowed_dist:
            best_son_node = now_node.expand()

            all_nodes += now_node.son_nodes
            now_node = best_son_node

            if now_node.iter_count > 100:
                break  # 防止实验里面跑太久

        # TODO: 接近 estimated_n_gates要进行剪枝, 同层要是做不到接近最优的node.after_optimize_dist / node.estimated_n_gates就会被剪枝
        solution_nodes.append(now_node)

        # 这样只会找最前面的一层的
        best_node: ApproaxitationNode = None
        best_reward = 0
        for node in all_nodes:
            if len(node.son_nodes) == 0:
                if best_reward < (node.after_optimize_dist / node.estimated_n_gates):  # 单位距离内需要门最少的
                    best_reward = (node.after_optimize_dist / node.estimated_n_gates)
                    best_node = node

        now_node = best_node

    if len(solution_nodes) == 0:
        return None, {}

    solutions = [
        node.get_circuit()
        for node in solution_nodes
    ]
    n_gates = [
        circuit.n_gates
        for circuit in solutions
    ]

    return solutions[int(np.argmin(n_gates))]


'''还不支持backend的进一步分解'''
'''TODO: 现在的depth很大'''

def decompose(target_U, allowed_dist, backend: Backend, max_n_solutions=1, multi_process=True, logger_level=logging.WARNING) -> Circuit:

    logger = logging.getLogger(__name__)
    logger.setLevel(logger_level)

    np.random.RandomState()

    n_qubits = int(math.log2(len(target_U)))

    assert n_qubits >= 4  # TODO: 小于4调用Qfast

    # 配置
    max_n_solutions = 1

    solutions: list[Circuit] = decompose_to_small_unitaries(target_U, n_candidates=n_qubits, backend=backend,
                                                                       allowed_dist=allowed_dist, multi_process=multi_process, logger=logger,)
    solutions = solutions[:max_n_solutions]
    
    # multi_process = False
    for candidate in solutions:
        for layer in candidate:
            for gate in layer:
                gate_qubits = gate['qubits']
                unitary_params = gate['params']

                unitary_params = reshape_unitary_params(
                    unitary_params, len(gate_qubits))
                gate_unitary = params_to_unitary(unitary_params)

                suub_backend = FullyConnectedBackend(n_qubits=len(
                    gate['qubits']), basis_single_gates=['u'], basis_two_gates=['crz'])

                kwargs = {
                    'target_U': gate_unitary,
                    'backend': suub_backend,
                    'allowed_dist': allowed_dist * 2,
                    'multi_process': multi_process,
                    'n_candidates': 10,
                    'logger': logger,
                }
                # gate['result'] = decompose_to_gates(**kwargs)
                if multi_process:
                    gate['result'] = decompose_to_gates_remote.remote(**kwargs)
                else:
                    gate['result'] = decompose_to_gates(**kwargs)

    finetune_futures = []
    for candidate in tqdm(solutions):
        circuit = []
        for layer in candidate:
            for gate in layer:
                gate_qubits = gate['qubits']
                gate['result'] = wait(gate['result'])

                gate_circuit = gate['result']

                for _layer in gate_circuit:
                    for _gate in _layer:
                        _gate['qubits'] = [gate_qubits[_qubit] for _qubit in _gate['qubits']]

                circuit += gate_circuit

        solution_dist = matrix_distance_squared(
            circuit_to_matrix(circuit, n_qubits), target_U)

        kwargs = {
            'n_qubits': n_qubits,
            'circuit': circuit,
            'U': target_U,
            'lr':max([solution_dist / 5, 1e-2]),
            'max_epoch': 2000,
            'unchange_tol': 1e-11,
            'n_iter_unchange': 200,
            'allowed_dist': allowed_dist,
            'verbose': False,
        }
        if multi_process:
            finetune_futures.append(find_parmas_remote.remote(**kwargs))
        else:
            finetune_futures.append(find_parmas.remote(**kwargs))

    finetune_dists = []
    finetune_solutions = []
    for circuit, solution_dist in wait(finetune_futures):
        finetune_solutions.append(circuit)
        finetune_dists.append(solution_dist)

    n_gate_solutions = [
        len([gate for layer in synthesized_solution for gate in layer])
        for synthesized_solution in finetune_solutions
    ]

    return finetune_solutions[int(np.argmin(n_gate_solutions))]

@ray.remote
def find_parmas_remote(*args, **kargs):
    return find_parmas(*args, **kargs)


@ray.remote
def decompose_to_small_unitaries_remote(*args, **kargs):
    return decompose_to_small_unitaries(*args, **kargs)


@ray.remote
def decompose_to_gates_remote(*args, **kargs):
    return decompose_to_gates(*args, **kargs)


@ray.remote
def optimize_remote(*args, **kargs):
    return optimize(*args, **kargs)
