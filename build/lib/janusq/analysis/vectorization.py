from functools import reduce
import logging
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import ray
from qiskit.circuit import QuantumCircuit, Instruction
from data_objects.circuit import Circuit, Gate, qiskit_to_circuit

from tools.saver import check_and_create_folder, dump, load

from data_objects.backend import Backend
from tools.ray_func import batch, wait, map

# TODO: path的结构替换成别的样子


def instruction2str(instruction):
    if isinstance(instruction, Instruction):
        qubits = [qubit.index for qubit in instruction.qubits]
        op_name = instruction.operation.name
    else:
        qubits = list(instruction['qubits'])
        qubits.sort()
        op_name = instruction['name']

    return f'{op_name},{",".join([str(_) for _ in qubits])}'


def extract_device(gate):
    if len(gate['qubits']) == 2:
        return tuple(sorted(gate['qubits']))
    else:
        return gate['qubits'][0]


class Step():
    '''A step contains (source gate, edge_type, target gate)'''

    def __init__(self, source, edge, target):
        self.source = instruction2str(source)
        self.edge = edge
        self.target = instruction2str(target)

    def __hash__(self): return hash(str(self))

    def __str__(self):
        if self.edge == 'loop':
            return str(self.target)
        else:
            return f'{self.edge}-{self.target}'

# TODO: there should be a step factory in the future to reduce the memory cost

class StepFactory():
    _hash_key = {}

    @staticmethod
    def create(gate):
        return Step()


class Path():
    '''A path consists of a list of step'''

    def __init__(self, steps):
        self.steps = steps
        self._path_id = str(self)

    def add(self, step):
        steps = list(self.steps)
        steps.append(step)
        return Path(steps)

    def __hash__(self): return hash(self._path_id)

    def __str__(self): return '-'.join([str(step) for step in self.steps])


def _bfs_walk(traveled_paths: set, traveled_gates: list, path, circuit: Circuit, now_gate: Gate, head_gate: dict, adjlist: dict, n_steps: int,
              n_walks: int, directions: list):
    if len(traveled_paths) > n_walks:
        return

    if n_steps <= 0:
        return

    now_layer = now_gate.layer_index
    parallel_gates = circuit[now_layer]
    former_gates = [] if now_layer == 0 else circuit[now_layer - 1]  # TODO: 暂时只管空间尺度的
    later_gates = [] if now_layer == len(
        circuit) - 1 else circuit[now_layer + 1]

    candidates = []
    if 'parallel' in directions:
        candidates += [('parallel', gate)
                       for gate in parallel_gates if gate != now_gate]
    if 'former' in directions:
        candidates += [('former', gate) for gate in former_gates]
    if 'next' in directions:
        candidates += [('next', gate) for gate in later_gates]

    ''' 对于gate只能游走到到相邻比特的门上 (adjlist)'''
    candidates = [
        (step_type, candidate)
        for step_type, candidate in candidates
        if candidate not in traveled_gates and
        any([(q1 in adjlist[q2] or q1 == q2) for q2 in now_gate['qubits'] for q1 in candidate['qubits']]) and
        any([(q1 in adjlist[q2] or q1 == q2)
            for q2 in traveled_gates[0]['qubits'] for q1 in candidate['qubits']])
    ]

    for step_type, next_gate in candidates:
        path_app = deepcopy(path)
        path_app = path_app.add(Step(now_gate, step_type, next_gate))
        path_id = path_app._path_id
        if path_id not in traveled_paths:
            traveled_paths.add(path_id)
        traveled_gates.append(next_gate)
        _bfs_walk(traveled_paths, traveled_gates, path_app, circuit, next_gate,
                  head_gate, adjlist, n_steps - 1, n_walks, directions)
        traveled_gates.remove(next_gate)


def walk_from_gate(circuit_info, head_gate, n_walks, n_steps, adjlist,
                   directions=('parallel', 'former', 'next')) -> set:
    traveled_paths = set()

    first_step = Path([Step(head_gate, 'loop', head_gate)])
    traveled_gates = [head_gate]
    _bfs_walk(traveled_paths, traveled_gates, first_step, circuit_info, head_gate, head_gate, adjlist, n_steps, n_walks,
              directions)

    op_qubits_str = instruction2str(head_gate)
    traveled_paths.add(op_qubits_str)

    return traveled_paths


def walk_on_circuit(circuit: Circuit, n_steps: int, n_walks: int, adjlist: dict,
                    directions=('parallel', 'former', 'next')):

    gate_paths = []
    for head_gate in circuit.gates:
        traveled_paths = walk_from_gate(circuit, head_gate, n_walks, n_steps, adjlist,
                                        directions)
        gate_paths.append(traveled_paths)
    return gate_paths

# meta-path只有三种 gate-parallel-gate, gate-former-gate, gate-next-gate
# n_steps: 定义了最大的步长


class RandomwalkModel():
    def __init__(self, n_steps, n_walks, backend: Backend, directions=('parallel', 'former', 'next')):
        '''
            n_walks: maximum number of random walks per gate
            n_steps: maximum step size per random walk
        '''
        self.model = None

        self.device_to_pathtable = defaultdict(
            dict)  # 存了路径(Path)到向量化后的index的映射
        # device 包括了coupler 和 qubit

        self.device_to_pathtable = defaultdict(
            dict)  # qubit -> path -> index
        self.device_to_reverse_pathtable = defaultdict(
            dict)  # qubit -> index -> path
        self.n_steps = n_steps
        self.n_walks = n_walks
        self.dataset = None

        self.backend = backend
        self.directions = directions
        self.n_qubits = backend.n_qubits

    def all_paths(self):
        return reduce(lambda a, b: list(a if isinstance(a, list) else a.keys()) + list(b.keys()), self.device_to_pathtable.values())
    
    def path_index(self, device, path):
        pathtable = self.device_to_pathtable[device]
        reverse_pathtable = self.device_to_reverse_pathtable[device]

        if path not in pathtable:
            pathtable[path] = len(pathtable)
            reverse_pathtable[pathtable[path]] = path

        return pathtable[path]

    def has_path(self, device, path):
        return path in self.device_to_pathtable[device]

    def train(self, circuits: list[Circuit], multi_process: bool = False, n_process: int = 10, remove_redundancy=True, is_filter_path=False, filter_path=None):
        logging.info(f'start random walk for {len(circuits)} circuits')

        backend: Backend = self.backend
        adjlist: dict = backend.adjlist
        n_steps = self.n_steps
        n_walks = self.n_walks

        paths_per_circuit: list[list[str]] = map(walk_on_circuit, circuits, multi_process, n_process, show_progress=True, n_steps=n_steps, n_walks=n_walks,
                                adjlist=adjlist, directions=self.directions)
        path_coexist_count = {}

        logging.info('count path')
        device_path_count: dict[str, dict] = {}
        for qubit in range(backend.n_qubits):
            device_path_count[qubit] = defaultdict(int)
        for coupling in backend.coupling_map:
            device_path_count[tuple(coupling)] = defaultdict(int)

        for circuit, paths_per_gate in zip(circuits, paths_per_circuit):
            for gate_index, paths in enumerate(paths_per_gate):
                paths = list(paths)
                device = extract_device(circuit.gates[gate_index])
                for path in paths:
                    device_path_count[device][path] += 1

        '''TODO: 很慢， 比特数大了就没法计算了'''
        redundant_paths = set()
        if remove_redundancy:
            for circuit, paths_per_gate in zip(circuits, paths_per_circuit):
                for gate_index, paths in enumerate(paths_per_gate):
                    paths = list(paths)

                    for i1, p1 in enumerate(paths):
                        for p2 in paths[i1+1:]:
                            if p1 not in path_coexist_count:
                                path_coexist_count[p1] = defaultdict(int)
                            path_coexist_count[p1][p2] += 1

            for device, path_count in device_path_count.items():
                logging.info(
                    f'{device} has {len(path_count)} paths before removing redundancy')
                paths = list(path_count.keys())
                for i1, p1 in enumerate(paths):
                    if p1 in redundant_paths:
                        continue
                    for p2 in paths[i1+1:]:
                        if p2 in redundant_paths or p1 not in path_coexist_count or path_coexist_count[p1][p2] == 0:
                            continue
                        if path_coexist_count[p1][p2] / path_count[p1] > 0.9 and path_count[p1] / path_count[p2] > 0.9 and path_count[p2] / path_count[p1] > 0.9:
                            if path_count[p1] > path_count[p2]:
                                redundant_paths.add(p1)
                            else:
                                redundant_paths.add(p2)

        # unusual paths are not added to the path table
        for device in device_path_count:
            for path, count in device_path_count[device].items():
                if count >= 10 and path not in redundant_paths:
                    self.path_index(device, path)

        logging.info(
            f'device size after random walk = {len(self.device_to_pathtable)}')

        self.max_table_size = 0
        for device, pathtable in self.device_to_pathtable.items():
            logging.info(f"{device}'s path table size = {len(pathtable)}")
            if len(pathtable) > self.max_table_size:
                self.max_table_size = len(pathtable)

        vecs_per_circuit = []
        for circuit, paths_per_gate in zip(circuits, paths_per_circuit):
            gate_vecs = []

            for gate, gate_paths in zip(circuit.gates, paths_per_gate):
                device = extract_device(gate)
                path_indices = [
                    self.path_index(device, path)
                    for path in gate_paths
                    if self.has_path(device, path)
                ]
                path_indices.sort()

                vec = np.zeros(self.max_table_size, dtype=np.int8)
                if len(path_indices) != 0:
                    vec[np.array(path_indices)] = 1.
                gate_vecs.append(vec)

            vecs_per_circuit.append(np.array(gate_vecs))

        return vecs_per_circuit   # paths_per_circuit

    def vectorize(self, circuit: Circuit, target_gates: list[int] = None) -> np.ndarray:
        if not isinstance(circuit, Circuit):
            assert target_gates is None
            return map(self.vectorize, circuit, False)
        
        if target_gates is None:
            target_gates = circuit.gates

        adjlist = self.backend.adjlist

        vecs = []    
        for gate in target_gates:
            paths = walk_from_gate(circuit, gate, self.n_walks, self.n_steps, adjlist,
                                   directions=self.directions)
            device = extract_device(gate)
            path_indices = [
                self.path_index(device, path_id) 
                for path_id in paths
                if self.has_path(device, path_id)
            ]
            path_indices.sort()

            vec = np.zeros(self.max_table_size, dtype=np.float32)
            if len(path_indices) != 0:
                vec[np.array(path_indices)] = 1.
                
            vecs.append(vec)

        return np.array(vecs)

    '''
        ==========================================================================
            The following three functions (extract_paths_from_vec, parse_gate_str, reconstruct) 
        is used to reconstruct circuits with the gate vectors.
        ==========================================================================
    '''

    def extract_paths_from_vec(self, device, gate_vector: np.array) -> list:
        inclued_path_indexs = np.argwhere(gate_vector == 1)[:, 0]
        paths = [
            self.device_to_pathtable[device][index]
            for index in inclued_path_indexs
        ]
        return paths

    @staticmethod
    def parse_gate_str(gate_str):
        elms = gate_str.split(',')
        gate = {
            'name': elms[0],
            'qubits': [int(qubit) for qubit in elms[1:]]
        }
        if elms[0] in ('rx', 'ry', 'rz'):
            gate['params'] = np.random.rand(1) * 2 * np.pi
        if elms[0] in ('u'):
            gate['params'] = np.random.rand(3) * 2 * np.pi
        elif elms[0] in ('cx', 'cz'):
            gate['params'] = []

        return gate

    def reconstruct(self, device, gate_vector: np.array) -> list:
        paths = self.extract_paths_from_vec(device, gate_vector)

        def add_to_layer(layer, gate):
            for other_gate in layer2gates[layer]:
                if instruction2str(other_gate) == instruction2str(gate):
                    return
            layer2gates[layer].append(gate)
            return

        head_gate = {
            'name': 'u',
            'qubits': [random.randint(0, self.n_qubits-1)],
            'params':  np.ones((3,)) * np.pi * 2
        }
        layer2gates = [
            list()
            for _ in range(self.n_steps * 2 + 1)
        ]
        head_layer = self.n_steps

        add_to_layer(head_layer, head_gate)

        for path in paths:
            now_layer = head_layer

            elms = path.split('-')
            if len(elms) == 1:
                head_gate.update(self.parse_gate_str(elms[0]))
                head_gate['params'] *= 3
            else:
                for index in range(0, len(elms), 2):
                    relation, gate_info = elms[index:index + 2]
                    if relation == 'next':
                        now_layer += 1
                    elif relation == 'former':
                        now_layer -= 1
                    add_to_layer(now_layer, self.parse_gate_str(gate_info))

        layer2gates = [
            layer
            for layer in layer2gates
            if len(layer) > 0
        ]

        return layer2gates

    @staticmethod
    def load(name):
        return load(name)

    def save(self, name):
        dump(name, self)
