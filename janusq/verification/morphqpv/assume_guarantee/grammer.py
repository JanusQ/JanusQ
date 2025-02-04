import numpy as np
from .solver import Solver,SGDSolver,LGDSolver
from .sample import build_relation
from .predicate import Predicate
from typing import Iterable
from .config import Config
class CircuitTape:
    def __init__(self, *args, **kwargs):
        self.circuit_data = []
        return
    def update(self):
        return
    """ the following methods are used to add gates to the circuit_data """
    def x(self, qubits: list):
        self.circuit_data.append({'name': 'x', 'qubits': qubits})
        return
    def y(self, qubits: list):    
        self.circuit_data.append({'name': 'y', 'qubits': qubits})
        return
    def z(self, qubits: list):
        self.circuit_data.append({'name': 'z', 'qubits': qubits})
        return
    def h(self, qubits: list):
        self.circuit_data.append({'name': 'h', 'qubits': qubits})
        return
    def rx(self, qubits: list, params: list):
        """add a rx gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angles for each qubit, unit: rad
        """
        self.circuit_data.append({'name': 'rx', 'qubits': qubits, 'params': params})
        return
    def ry(self, qubits: list, params: list):
        """add a ry gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angles for each qubit, unit: rad
        """
        self.circuit_data.append({'name': 'ry', 'qubits': qubits, 'params': params})
        return
    def rz(self, qubits: list, params: list):
        """add a rz gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angle
        """
        self.circuit_data.append({'name': 'rz', 'qubits': qubits, 'params': params})
        return
    def s(self, qubits: list):
        """add a s gate to the circuit_data
        Args:
            qubits (list): the qubits to apply the gate
        """
        self.circuit_data.append({'name': 's', 'qubits': qubits})
        return
    def cnot(self, qubits: list):
        """add a cnot gate to the circuit_data

        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({'name': 'cx', 'qubits': qubits})
        return
    def cx(self, qubits: list):
        """add a cnot gate to the circuit_data


        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({'name': 'cx', 'qubits': qubits})
        return
    def cz(self, qubits: list):
        """add a cz gate to the circuit_data

        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({'name': 'cz', 'qubits': qubits})
        return
    def swap(self, qubits: list):
        """add a swap gate to the circuit_data


        Args:
            qubits (list): [qubit1, qubit2]
        """
        self.circuit_data.append({'name': 'swap', 'qubits': qubits})
        return
    def t(self, qubits: list):
        self.circuit_data.append({'name': 't', 'qubits': qubits})
        return
    def tdg(self, qubits: list):
        self.circuit_data.append({'name': 'tdg', 'qubits': qubits})
        return
    def u1(self, qubits: list, params: list):
        """ add a u1 gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the phase angles for each qubit, unit: rad
        """
        self.circuit_data.append({'name': 'u1', 'qubits': qubits, 'params': params})
        return
    def u2(self, qubits: list, params: list):
        """add a u2 gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the [phi, lambda] angles for each qubit, unit: rad
        """
        self.circuit_data.append({'name': 'u2', 'qubits': qubits, 'params': params})
        return
    def u3(self, qubits, params):
        self.circuit_data.append({'name': 'u3', 'qubits': qubits, 'params': params})
        return
    def unitary(self, qubits, params):
        self.circuit_data.append({'name': 'unitary', 'qubits': qubits, 'params': params})
        return
    def mcx(self, qubits):
        self.circuit_data.append({'name': 'mcx', 'qubits': qubits})
        return
    def mcp(self, qubits):
        self.circuit_data.append({'name': 'mcp', 'qubits': qubits})
        return
    
    def to_qiskit_circuit(self):
        """convert the circuit_data to qiskit circuit"""
        return
    def from_qiskit_circuit(self, qiskit_circuit):
        """convert the qiskit circuit to circuit_data"""
        return
    def to_layer_circuit(self, circuit_data):
        """convert the circuit_data to layer circuit"""
        layer_circuit = [[]]
        for gate in circuit_data:
            is_new_layer = False
            for qubit in gate['qubits']:
                if qubit in [gate['qubits'] for gate in layer_circuit[-1]]:
                    is_new_layer = True
                    break
            if is_new_layer:
                layer_circuit.append([gate])
            else:
                layer_circuit[-1].append(gate) 
        return layer_circuit


class MorphQC(CircuitTape):
    def __init__(self,circuit_data: list = None,config=None,**kwargs):
        """use circuit_data to initialize the MorphQC """
        super().__init__(**kwargs)
        if config:
            self.config = config
        else:
            self.config = Config()
        if circuit_data is None:
            self.circuit_data = []
        else:
            if isinstance(circuit_data, list):
                self.circuit_data = circuit_data
            else:
                raise ValueError('circuit_data must be a list')
        self.tracepoints = []
        self.clean()
        self.update()
        
    def add_tracepoint(self, *args, **kwargs):
        '''add a tracepoint to the circuit_data'''
        self.circuit_data.append({'name': 'tracepoint', 'qubits': args})
        return
    
    def clean(self):
        '''clean the assume and gurrantee points'''
        self.assume_points = []
        self.gurrantee_points = []
        self.verify_point_idx = []
    def update(self):
        '''update the circuit blocks by tracepoints'''
        self.circuitBlocksByTracePoints()
        return
    def circuitBlocksByTracePoints(self, *args, **kwargs):
        '''return a list of circuit blocks'''
        self.tracepoint_idxes_in_layer = []
        for idx,gate in enumerate(self.circuit_data):
            if 'tracepoint' == gate['name']:
                self.tracepoints.append(gate)
                self.tracepoint_idxes_in_layer.append(idx)
        return
    def assume(self,tracepoint_idxes: Iterable[int],predicate: Predicate,*args):
        ''' assume the predicate statement in tracepoint_idxes
        Args:
            tracepoint_idxes (List[int]): the index of the tracepoint, 0 for the input tracepoint, 1 for the first tracepoint, 2 for the second tracepoint, etc.
            predicate (Predicate): the predicate function to be assumed
            args: the arguments of the predicate function
        '''
        if isinstance(tracepoint_idxes, int):
            tracepoint_idxes = [tracepoint_idxes]
        self.assume_points.append((tracepoint_idxes, predicate, args))
        for tracepoint_idx in tracepoint_idxes:
            if tracepoint_idx not in self.verify_point_idx:
                self.verify_point_idx.append(tracepoint_idx)
        return
    
    def guarantee(self,tracepoint_idxes: Iterable[int],predicate: Predicate,*args):
        ''' guarantee the predicate statement in tracepoint_idxes
        Args:
            tracepoint_idxes (List[int]): the index of the tracepoint, 0 for the input tracepoint, 1 for the first tracepoint, 2 for the second tracepoint, etc.
            predicate (Predicate): the predicate function to be assumed
            args: the arguments of the predicate function
        '''
        if isinstance(tracepoint_idxes, int):
            tracepoint_idxes = [tracepoint_idxes]
        self.gurrantee_points.append((tracepoint_idxes, predicate,args))
        for tracepoint_idx in tracepoint_idxes:
            if tracepoint_idx not in self.verify_point_idx:
                self.verify_point_idx.append(tracepoint_idx)
        return
    
    def get_solver(self):
        '''get the solver'''
        if self.config.solver == 'sgd':
            return SGDSolver(**self.config.__dict__)
        elif self.config.solver == 'lgd':
            return LGDSolver(**self.config.__dict__)
        else:
            return Solver()
    def verify(self):
        '''verify the assume and gurrantee
        Returns:
            results: the results of the verification
            {
                'assume': self.assume_points,
                'gurrantee': self.gurrantee_points,
                'verify': {
                    'optimal_input_state': the optimal input state,
                    "optimal_gurrantee_value": the optimal gurrantee value,
                    "is_assume_satisfied" : the optimal assume value
                }
            }
        '''
        self.verify_point_idx.sort()
        self.update()
        solver = self.get_solver()
        if len(self.verify_point_idx) == 0:
            return
        for idx in self.verify_point_idx:
            if idx == 0:
                continue
            verify_block = self.circuit_data[:self.tracepoint_idxes_in_layer[idx]]
            verify_block = self.to_layer_circuit(verify_block)
            relation = build_relation(verify_block,self.tracepoints[0]['qubits'], self.tracepoints[idx]['qubits'],**self.config.__dict__)
            solver.add_relation(0, idx, relation)

        for idxes, predicate, args in self.assume_points:
            solver.add_constraint(idxes,predicate,args)
        for idxes, predicate, args in self.gurrantee_points:
            solver.add_objective(idxes,predicate,args)
        results = solver.solve()
        self.assertion = {
            'assume': self.assume_points,
            'gurrantee': self.gurrantee_points,
            'verify': results
        }
        self.clean()
        return results
    
    def __enter__(self):

        return self
    
    def __exit__(self,exc_type, exc_val, exc_tb):

        return self.verify()

