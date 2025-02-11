'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 06:05:01
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-18 07:14:18
FilePath: /JanusQ/janusq/data_objects/backend.py
Description: this module define several kind of backends

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
import copy
import itertools as it
import math
import time
import random
from collections import defaultdict

DEFAUT_SINGLE_QGATES = ['u']
DEFAULT_DOUBLE_QGATES = ['cx']


class Backend():
    '''
    topology: {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]})
    coupling_map: [[0, 1], [1, 2], [3, 4], [5, 8], [0, 3], [1, 4], [6, 7], [4, 5], [3, 6], [2, 5], [4, 7], [7, 8]]
    neigh_info: {0: [1, 3], 1: [0, 3], 2: [1], 3: [1, 4, 7], 4: [1, 3, 5, 7], 5: [1, 4, 7], 6: [7], 7: [5, 8], 8: [5, 7]})  
    '''
    
    def __init__(self, n_qubits: int, topology: dict = None, adjlist: dict=None, coupling_map: list = None, basis_single_gates: list = None,
                 basis_two_gates: list = None, single_qubit_gate_time: int=30, two_qubit_gate_time: int=60,):
        '''
        description: 
        param {int} n_qubits: the number of qubit
        param {dict} topology: the topology of qubits, the list of value are qubits connected with the key qubit
        param {dict} adjlist: the topology of qubits, the list of value are qubits connected with the key qubit
        param {list} coupling_map: the list of couplers
        param {list} basis_single_gates: support single gates
        param {list} basis_two_gates: support two qubit gates
        param {int} single_qubit_gate_time: the exection time of single gate
        param {int} two_qubit_gate_time: the exection time of two qubit gate
        '''

    
        self.n_qubits = n_qubits
        self.involvod_qubits = list(range(n_qubits))

        if topology is None:
            topology = {
                q1: [q2 for q2 in range(n_qubits) if q1 != q2]
                for q1 in range(n_qubits)
            }

        self.topology = topology
        if coupling_map is None:
            self.coupling_map = _topology_to_coupling_map(topology)
        else:
            self.coupling_map = coupling_map
        # self._coupling_map = [tuple(elm) for elm in coupling_map]
            
        self._true_coupling_map = list(self.coupling_map)

        if adjlist is None:
            self.adjlist = copy.deepcopy(topology)
        else:
            self.adjlist = adjlist

        if basis_single_gates is None:
            basis_single_gates = DEFAUT_SINGLE_QGATES

        if basis_two_gates is None:
            basis_two_gates = DEFAULT_DOUBLE_QGATES

        self.basis_single_gates = basis_single_gates
        self.basis_two_gates = basis_two_gates
        self.basis_gates = self.basis_single_gates + self.basis_two_gates

        self.single_qubit_gate_time = single_qubit_gate_time  # ns
        self.two_qubit_gate_time = two_qubit_gate_time  # ns
        
        self.cache = {}


        
    def get_subgraph(self, location):
        """Returns the sub_coupling_graph with qubits in location."""
        subgraph = []
        for q0, q1 in self.coupling_map:
            if q0 in location and q1 in location:
                subgraph.append((q0, q1))
        return subgraph

    def get_sub_backend(self, sub_qubits):
        sub_backend = copy.deepcopy(self)
        sub_backend.topology = {
            qubit: [] if qubit not in sub_qubits else [
                connect_qubit for connect_qubit in connect_qubits if connect_qubit in sub_qubits]
            for qubit, connect_qubits in self.topology.items()
        }
        sub_backend.coupling_map = _topology_to_coupling_map(
            sub_backend.topology)
        sub_backend.involvod_qubits = list(sub_qubits)
        return sub_backend

    def get_connected_qubit_sets(self, n_qubit_set):
        """
        Returns a list of qubit sets that complies with the topology.
        """

        assert n_qubit_set < self.n_qubits and n_qubit_set > 0, (n_qubit_set, self.n_qubits)
        
        if n_qubit_set in self.cache:
            return self.cache[n_qubit_set]

        locations = []

        for group in it.combinations(range(self.n_qubits), n_qubit_set):
            # Depth First Search
            seen = set([group[0]])
            frontier = [group[0]]

            while len(frontier) > 0 and len(seen) < len(group):
                for q in group:
                    if frontier[0] in self.topology[q] and q not in seen:
                        seen.add(q)
                        frontier.append(q)

                frontier = frontier[1:]

            if len(seen) == len(group):
                locations.append(group)

        self.cache[n_qubit_set] = locations
        return locations

    def __eq__(self, other):
        return self.n_qubits == other.n_qubits



class FullyConnectedBackend(Backend):
    '''
    fully connected backend, qubits are connected with other qubits

    Example:
    0-1-2
    |___|

    '''
    def __init__(self, n_qubits:int, **kwargs):
        '''
        param {int} n_qubits: number of qubits
        '''
        topology = {
            q1: [q2  for q2 in range(n_qubits) if q1 != q2]
            for q1 in range(n_qubits)
        }
        Backend.__init__(self, n_qubits, topology=topology, **kwargs)
        return



class GridBackend(Backend):
    '''
    Grid-like backend, qubit is organized into a grid shape

    Example:
    0-1-2
    | | |
    3-4-5
    | | |
    6-7-8
    '''
    def __init__(self, n_columns:int, n_rows:int, dist_threadhold:int = 100, **kwargs):
        '''
        param {int} n_columns: number of columns
        param {int} n_rows: number of rows
        param {int} dist_threadhold: maximum distances of neighboring qubits in the random walk
        '''
        topology = defaultdict(list)

        for x in range(n_columns):
            for y in range(n_rows):
                qubit = x * n_rows + y
                for neigh_x in range(x - 1, x + 2):
                    neigh_y = y
                    if neigh_x < 0 or neigh_x >= n_columns or x == neigh_x:
                        continue
                    neigh_qubit = neigh_x * n_rows + neigh_y
                    topology[qubit].append(neigh_qubit)

                for neigh_y in range(y - 1, y + 2):
                    neigh_x = x
                    if neigh_y < 0 or neigh_y >= n_rows or y == neigh_y:
                        continue
                    neigh_qubit = neigh_x * n_rows + neigh_y
                    topology[qubit].append(neigh_qubit)

        for qubit, coupling in topology.items():
            coupling.sort()

        n_qubits = n_columns * n_rows
        adjlist = defaultdict(list)
        
        for q1 in range(n_qubits):
            x1, y1 = q1//n_rows, q1%n_rows
            for q2 in range(n_qubits):
                if q1 == q2: continue
                x2, y2 = q2//n_rows, q2%n_rows
                if math.sqrt((x1-x2)**2 + (y1-y2)**2) < dist_threadhold:
                    adjlist[q1].append(q2)

        Backend.__init__(self, n_qubits, topology = topology, adjlist = adjlist, **kwargs)
        

class LinearBackend(Backend):
    def __init__(self, n_qubits, dist_threadhold = 100, **kwargs):
        """
        Initialize a linear quantum backend object.

        Parameters
        ----------
        n_qubits : int
            Total number of qubits in the backend.

        dist_threadhold : int, optional
            Distance threshold between qubits. Default is 100.

        **kwargs : dict
            Other parameters passed to the constructor of the parent class Backend.

        Attributes
        ----------
        topology : dict
            The topology of the quantum backend, representing the connectivity between qubits.

        adjlist : dict
            The adjacency list of the quantum backend, representing the connectivity of each qubit to others.

        Raises
        ------
        ValueError
            If dist_threadhold is not a positive integer.

        """
        # Build topology: connectivity between each qubit and its neighbors
        topology = {
            q1: [q2 for q2 in [q1-1, q1+1] if q2 >= 0 and q2 < n_qubits]
            for q1 in range(n_qubits)
        }
        # Build adjacency list: establish connections between qubits within dist_threadhold
        adjlist = {
        q1: [
            q2 for q2 in range(n_qubits)
            if q1 != q2 and (q1-q2)**2 <= dist_threadhold**2
        ]
        for q1 in range(n_qubits)
    }
        
        # Call the constructor of the parent class Backend and pass topology and adjlist as parameters
        Backend.__init__(self, n_qubits, topology = topology, adjlist = adjlist, **kwargs)


def _topology_to_coupling_map(topology: dict) -> list:
    coupling_map = set()
    for qubit, coupling in topology.items():
        for neighbor_qubit in coupling:
            coupling = [qubit, neighbor_qubit]
            coupling.sort()
            coupling_map.add(tuple(coupling))
    return [
        list(coupling)
        for coupling in coupling_map
    ]


def devide_qubits(topology: dict, max_qubits: int):
    '''
        device qubits into non-overlapping groups with max_qubit
    '''
    qubits = topology.keys()
    trevel_node, devide_qubits = [], []

    random.seed(time.time())
    while len(trevel_node) != len(qubits):
        sub_qubits = []
        head = random.choice(list(qubits-trevel_node))
        
        fommer_step = topology[head]
        trevel_node.append(head)
        sub_qubits.append(head)
        t = 1
        while t < max_qubits:
            new_fommer_step = []
            for fommer_qubit in fommer_step:
                if t == max_qubits:
                    break
                if fommer_qubit in trevel_node:
                    continue
                sub_qubits.append(fommer_qubit)
                trevel_node.append(fommer_qubit)
                new_fommer_step+= topology[fommer_qubit]
                t += 1
            if len(new_fommer_step) == 0:
                break
            fommer_step = list(set(new_fommer_step))
            if head in fommer_step:
                fommer_step.remove(head)
            
        sub_qubits.sort()
        devide_qubits.append(sub_qubits)

    return devide_qubits


def get_sub_backend(backend: Backend, max_qubit: int, devide_qubits = None):
    ret_backend = copy.deepcopy(backend)
    if not devide_qubits:
        devide_qubits = devide_qubits(ret_backend.topology,  max_qubit)
    ret_backend.devide_qubits = devide_qubits
    coupling_map = copy.deepcopy(ret_backend.coupling_map)
    for e1, e2 in coupling_map:
        for i in range(len(devide_qubits)):
            if (e1 in devide_qubits[i] and e2 not in devide_qubits[i]) or (e1 not in devide_qubits[i] and e2 in devide_qubits[i]):
                ret_backend.coupling_map.remove([e1, e2])
                break
    return ret_backend
