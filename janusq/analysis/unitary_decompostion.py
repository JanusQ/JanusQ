'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 03:33:02
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-19 02:51:09
FilePath: /JanusQ/janusq/analysis/unitary_decompostion.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 03:33:02
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-19 02:06:14
FilePath: /JanusQ/janusq/analysis/unitary_decompostion.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
from collections import defaultdict
import copy
import math
import time
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

from janusq.analysis.vectorization import RandomwalkModel, extract_device
from janusq.data_objects.backend import Backend, FullyConnectedBackend
from janusq.data_objects.circuit import Circuit, Layer, circuit_to_qiskit, qiskit_to_circuit
from janusq.tools.optimizer import OptimizingHistory
from janusq.tools.ray_func import wait, map
import inspect
import os



class PCA():
    def __init__(self, X, k=None, max_k=None, reduced_prop=None) -> None:
        '''
        description: reduce vector dimensions
        param {*} X: data to reduce
        param {*} max_k: top k max eigen value
        param {*} reduced_prop: the propotion of demension to reduce
        '''
        X = np.concatenate([m.reshape((-1, m.shape[-1])) for m in X], axis=0)
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        C = np.cov(X_centered.T)

        eigvals, eigvecs = jnp.linalg.eig(C)

        sorted_indices = jnp.argsort(eigvals)[::-1]
        sorted_eigen_values = eigvals[sorted_indices]

        sum_eigen_values = jnp.sum(sorted_eigen_values)

        if reduced_prop is not None:
            k = 0
            target_eigen_values = sum_eigen_values * reduced_prop
            accumulated_eigen_value = 0
            for eigen_value in sorted_eigen_values:
                accumulated_eigen_value += eigen_value
                k = k + 1
                if accumulated_eigen_value > target_eigen_values or (max_k is not None and k >= max_k):
                    break

        if k is not None:
            accumulated_eigen_value = 0
            for eigen_value in sorted_eigen_values[:k]:
                accumulated_eigen_value += eigen_value
            reduced_prop = accumulated_eigen_value / sum_eigen_values

        # print('k =', k)
        # print('reduced_prop =', reduced_prop)

        self.k = k
        self.reduced_prop = reduced_prop
        self.V_k = eigvecs[:, sorted_indices[:k]]
        self.eigvecs = eigvecs
        self.sorted_indices = sorted_indices[:k]
        self.X_mean = X_mean
        pass

    def transform(self, X) -> jnp.array:
        reduced_matrices: jnp.array = vmap(pca_transform, in_axes=(0, None, None))(X, self.X_mean, self.V_k)
        return reduced_matrices.astype(jnp.float64)


@jax.jit
def pca_transform(m, X_mean, V_k):
    m_centered = m - X_mean[jnp.newaxis, :]
    m_reduced = jnp.dot(m_centered, V_k)
    q, r = jnp.linalg.qr(m_reduced)
    q = q.reshape(q.size)
    q = jnp.concatenate([q.imag, q.real], dtype=jnp.float64)
    return q



def random_params(circuit: Circuit):
    circuit = copy.deepcopy(circuit)
    for layer in circuit:
        for gate in layer:
            gate['params'] = np.random.rand(len(gate['params'])) * 2 * np.pi
    return circuit


def flatten(U: np.ndarray)->np.ndarray:
    U = U.reshape(U.size)
    return np.concatenate([U.real, U.imag])

class U2VModel():
    def __init__(self, upstream_model: RandomwalkModel, name=None):
        '''
        description: turn unitary to vector candidates
        param {RandomwalkModel} upstream_model:
        param {str} name: model name
        '''
        self.upstream_model = upstream_model
        self.backend = upstream_model.backend
        self.n_qubits = upstream_model.n_qubits
        self.name = name

        self.pca_model = None
        self.U_to_vec_model = None

    def construct_data(self, circuits, multi_process=False):
        '''
        description: use circuits' vecs and its unitary construct an U2V model 
        param {*} circuits: train dataset
        param {bool} multi_process: weather to enable multi=process
        '''
        n_qubits = self.n_qubits
        n_steps = self.upstream_model.n_steps

        def gen_data(circuit, n_qubits):
            device_gate_vecs, Us = [], []
            sub_circuits = []

            gate_vec = self.upstream_model.vectorize(circuit)
            for layer_index, layer in enumerate(circuit):
                for target_gate in layer:
                    if len(target_gate['qubits']) == 1:
                        continue

                    gate_vec = target_gate.vec
                    if len(gate_vec) == 1:
                        continue

                    device_gate_vecs.append([extract_device(target_gate), np.argwhere(gate_vec > 0).flatten()])
                    U = circuit_to_matrix(circuit, n_qubits)
                    Us.append(U)

                    sub_circuit = circuit[layer_index:]
                    sub_circuits.append(sub_circuit[: n_steps])
                    break

            return device_gate_vecs, Us, sub_circuits

        # @ray.remote
        # def gen_data_remote(circuit_info, n_qubits):
        #     return gen_data(circuit_info, n_qubits)

        # print('Start generating Us -> Vs, totoal', len(circuits))
        # futures = []
        # for index, circuit_info in enumerate(circuits):
        #     if multi_process:
        #         future = gen_data_remote.remote(circuit_info, n_qubits)
        #     else:
        #         future = gen_data(circuit_info, n_qubits)
        #     futures.append(future)
        # wait(futures, show_progress=True)
        
        futures = map(gen_data, circuits, multi_process, n_qubits = n_qubits)
        

        Vs, Us = [], []
        sub_circuits = []
        for index, future in enumerate(futures):
            Vs += future[0]
            Us += future[1]
            sub_circuits += future[2]

        Us = np.array(Us, dtype=np.complex128)  # [:100000]

        print('len(Us) = ', len(Us), 'len(gate_vecs) = ', len(Vs))

        return Us, Vs, sub_circuits

    def train(self, data, n_candidates=15, reduced_prop=.6, max_k=100):
        print('Start construct U2VMdoel')
        start_time = time.time()

        Us, Vs, sub_circuits = data

        self.pca_model = PCA(Us, reduced_prop=reduced_prop, max_k=max_k)

        Us = self.pca_model.transform(Us)
        Us = [flatten(U) for U in Us]

        self.nbrs = NearestNeighbors(n_neighbors=n_candidates, n_jobs=-1).fit(Us)  # algorithm='ball_tree',
        self.Vs = Vs
        self.sub_circuits = sub_circuits

        print(f'Finish construct U2VMdoel, costing {time.time() - start_time}s')

    def choose(self, U, verbose=False):
        nbrs: NearestNeighbors = self.nbrs
        upstream_model = self.upstream_model
        Vs = self.Vs

        U = self.pca_model.transform(jnp.array([U]))[0]
        U = flatten(U)

        distances, indices = nbrs.kneighbors([U])
        distances, indices = distances[0], indices[0]

        candidates = []

        if len(indices) == 0: return []

        for index in indices:
            candidate = self.sub_circuits[index]
            candidate = [layer_gates for layer_gates in candidate if len(layer_gates) != 0]
            if len(candidate) == 0:
                continue
            candidates.append(candidate)

        return candidates


def recongnize_analysis():
    pass
config.update("jax_enable_x64", True)
current_dir = os.path.dirname(inspect.getfile(recongnize_analysis))

'''load data for decompostion'''
IPARAMS = {}
with open(os.path.join(current_dir, 'decomposition_data/identity_params.pkl'), 'rb') as file:
    IPARAMS = pickle.load(file)

RFS = {}  # redundancy-free candidate set
for n_qubits in range(4, 7):
    with open(os.path.join(current_dir, f'decomposition_data/{n_qubits}_crz_best.pkl'), 'rb') as file:
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
    q, r = jnp.linalg.qr(z)
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
        n_connect_qubits = len(connect_qubits)
        return [[{
            'name': 'unitary',
            'qubits': list(connect_qubits),
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

def circuit_to_matrix(circuit: Circuit, n_qubits, params=None) -> jax.numpy.array:
    '''
    description: compute unitary of circuit
    param {Circuit} circuit: target circuit
    param {int} n_qubits: numbrer of qubits
    param {*} params: kwargs
    return {np.ndarray} unitary
    '''
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
    # for _ in range(3):
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
    def __init__(self, target_U: np.array, former_circuit: Circuit, inserted_layers: list[Layer], iter_count: int, config: dict, former_node=None,u2v_model = None):
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
        self.u2v_model = u2v_model

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
        '''update reward'''
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

        canditate_layers += RFS[self.n_qubits]

        canditate_layers = remove_repeated_candiate(canditate_layers)

        futures = []
        for candidate_layer in canditate_layers:
            son_node = ApproaxitationNode(self.target_U, self.circuit,
                                          candidate_layer, self.iter_count+1, self.config, self)
            self.son_nodes.append(son_node)

            futures.append((son_node, son_node.optimize()))

        for son_node, future in wait(futures):
            son_node.update(*future)

        rewards = np.array([son_node.reward for son_node in self.son_nodes])
        if np.all(rewards < 0):
            self.logger.error(f'No improvement after inserting gates')

        best_son_node: ApproaxitationNode = self.son_nodes[np.argmax(rewards)]

        # self.logger.warning(
        #     f'iter_count= {self.iter_count} now_dist= {best_son_node.after_optimize_dist}, #gate = {best_son_node.estimated_n_gates} \n')
        
        logging.info(f'iter_count= {self.iter_count} now_dist= {best_son_node.after_optimize_dist}, {best_son_node.estimated_n_gates}')

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
            for _ in range(now_n_blocks): 
                candidate_qubits = random_choice(connected_qubit_sets)
                candidate += create_unitary_gate(candidate_qubits)
            candidates.append(candidate)

        candidates: list[Circuit] = remove_repeated_candiate(
            candidates) 

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
                # logging.warning(f'n_blocks of {n_qubits} qubits should be {n_qubits2n_blocks[n_qubits]}')
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

    # logging.info(f'Decompose to {min_n4q} 4-q unitaries. The number of solutions are {len(n_4q_to_solutions[min_n4q])}')
    return n_4q_to_solutions[min_n4q]


def decompose_to_gates(target_U, backend: Backend, allowed_dist=1e-5, multi_process=True, n_candidates=40, logger=None, u2v_model = None) -> list[Circuit]:
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
        'n_candidates': n_candidates,
        u2v_model: u2v_model,
    })
    root_node.after_optimize_dist = I_dist

    now_node = root_node

    max_solution_nodes = 1

    all_nodes: list[ApproaxitationNode] = []
    solution_nodes: list[ApproaxitationNode] = []
    now_node: ApproaxitationNode = root_node


    while len(solution_nodes) < max_solution_nodes:
        while now_node.after_optimize_dist >= allowed_dist:
            best_son_node = now_node.expand()

            all_nodes += now_node.son_nodes
            now_node = best_son_node

            if now_node.iter_count > 100:
                break

        solution_nodes.append(now_node)

        best_node: ApproaxitationNode = None
        best_reward = 0
        for node in all_nodes:
            if len(node.son_nodes) == 0:
                if best_reward < (node.after_optimize_dist / node.estimated_n_gates):
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


def decompose(target_U, allowed_dist, backend: Backend, max_n_solutions=1, multi_process=True, logger_level=logging.WARNING, u2v_model: U2VModel  = None) -> Circuit:
    '''
    description: 
    param {np.ndarray} target_U: target unitary
    param {float} allowed_dist: the distance allowed between target unitary and synthesis circuit
    param {Backend} backend: backend 
    param {*} max_n_solutions: select n candidates per iteration
    param {*} multi_process: weather to enbale multi-process
    param {*} logger_level: logging level
    param {U2VModel} u2v_model: use u2vmodel tot reduce search space of candidates
    return {Circuit}: synthesis circuit
    '''

    logger = logging.getLogger(__name__)
    logger.setLevel(logger_level)

    np.random.RandomState()

    n_qubits = int(math.log2(len(target_U)))

    assert n_qubits >= 4

    if n_qubits == 4:
        kwargs = {
            'target_U': target_U,
            'backend': backend,
            'allowed_dist': allowed_dist,
            'multi_process': multi_process,
            'n_candidates': 10,
            'logger': logger,
            "u2v_model": u2v_model,
        }
        return decompose_to_gates(**kwargs)
    
    # 配置
    max_n_solutions = 3

    solutions: list[Circuit] = decompose_to_small_unitaries(target_U, n_candidates=n_qubits, backend=backend,
                                                                       allowed_dist=min([0.1, allowed_dist*2]), multi_process=multi_process, logger=logger,)
    # TODO: check the code
    solutions = solutions[:max_n_solutions]
    
    for candidate in solutions:
        for layer in candidate:
            for gate in layer:
                gate_qubits = gate['qubits']
                unitary_params = gate['params']

                unitary_params = reshape_unitary_params(
                    unitary_params, len(gate_qubits))
                gate_unitary = params_to_unitary(unitary_params)

                sub_backend = FullyConnectedBackend(n_qubits=len(
                    gate['qubits']), basis_single_gates=['u'], basis_two_gates=['crz'])

                kwargs = {
                    'target_U': gate_unitary,
                    'backend': sub_backend,
                    'allowed_dist': allowed_dist * 2,
                    'multi_process': multi_process,
                    'n_candidates': 10,
                    'logger': logger,
                    "u2v_model": u2v_model,
                }

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

        print("solution_dist", solution_dist)
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

    return Circuit(finetune_solutions[int(np.argmin(n_gate_solutions))])

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
