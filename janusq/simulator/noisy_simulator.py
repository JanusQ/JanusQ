'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 03:33:02
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-18 08:14:13
FilePath: /JanusQ/janusq/simulator/noisy_simulator.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
import random
from collections import defaultdict
from numpy import pi
from qiskit import Aer
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info.analysis import hellinger_fidelity
from ..analysis.vectorization import RandomwalkModel, extract_device
from ..objects.circuit import Circuit, Gate, Layer, SeperatableCircuit
from ..objects.random_circuit import random_1q_layer
from qiskit_aer.noise import NoiseModel
from .readout_error_model import ReadoutErrorModel
from .gate_error_model import GateErrorModel
import numpy as np



class NoisySimulator():
    def __init__(self, backend, gate_error_model: GateErrorModel = None, readout_error_model: ReadoutErrorModel = None):
        """
        Initialize the NoisySimulator with the specified backend and error models.

        Args:
            backend (str): The backend used for simulation.
            gate_error_model (GateErrorModel): The gate error model used for simulation.
            readout_error_model (ReadoutErrorModel): The readout error model used for simulation.
        """
        self.gate_error_model: GateErrorModel = gate_error_model
        self.readout_error_model: ReadoutErrorModel = readout_error_model
        
        if gate_error_model is not None:
            assert gate_error_model.backend is backend
            self.vec_model: RandomwalkModel = gate_error_model.vec_model
            
        if readout_error_model is not None:
            assert readout_error_model.backend is backend
            
        self.backend = backend
        

        backend = self.backend

        self.qasm_simulator = Aer.get_backend('qasm_simulator')
        self.n_qubits = backend.n_qubits

    def obtain_circuit_fidelity(self, circuit: Circuit, n_samples=1000, circuit_reps=5):
        """
        Calculate the fidelity of the circuit executed on the noisy simulator.

        Args:
            circuit (Circuit): The quantum circuit to be executed.
            n_samples (int, optional): Number of samples for circuit execution. Default is 1000.
            circuit_reps (int, optional): Number of repetitions for circuit execution. Default is 5.

        Returns:
            float: Average fidelity of the noisy circuit execution.
            float: Average number of error paths encountered during execution.
        """
        n_qubits = circuit.n_qubits
        
        fidelities = []
        noisy_countses = []
        for _ in range(circuit_reps):
            layer_1q = random_1q_layer(
                n_qubits, circuit.operated_qubits, self.backend.basis_single_gates)

            simulated_circuit = Circuit(
                layer_1q, layer_1q.num_qubits) + circuit + Circuit(layer_1q.inverse(), layer_1q.num_qubits)

            true_result = self.execute_noise_free(simulated_circuit)
            
            noisy_result, n_error_paths = self.execute(
                simulated_circuit, n_samples, get_n_error_paths=True)
            fidelities.append(hellinger_fidelity(
                noisy_result, true_result))
            noisy_countses.append(n_error_paths)

        # print(sum(noisy_countses)/len(noisy_countses))
        return sum(fidelities)/len(fidelities), sum(noisy_countses)/len(noisy_countses)

    def obtain_seperable_circuit_fidelity(self, circuit: SeperatableCircuit, n_samples=1000, circuit_reps=5):
        """
        Calculate the fidelity of each sub-circuit in a separable circuit executed on the noisy simulator.

        Args:
            circuit (SeperatableCircuit): The separable quantum circuit to be executed.
            n_samples (int, optional): Number of samples for circuit execution. Default is 1000.
            circuit_reps (int, optional): Number of repetitions for circuit execution. Default is 5.

        Returns:
            list[float]: List of average fidelities for each sub-circuit.
            list[float]: List of average number of error paths encountered for each sub-circuit.
        """
        n_qubits = circuit.n_qubits

        gate_vecs = self.vec_model.vectorize(circuit)

        fidelity_list = []
        n_error_paths_list = []
        for subcircuit in circuit.seperatable_circuits:
            true_result = self.execute_noise_free(subcircuit)

            sub_fs = []
            sub_eps = []
            for _ in range(circuit_reps):
                layer_1q = random_1q_layer(
                    n_qubits, subcircuit.operated_qubits, self.backend.basis_single_gates)

                simulated_circuit = Circuit(
                    layer_1q, layer_1q.num_qubits) + subcircuit + Circuit(layer_1q.inverse(), layer_1q.num_qubits)

                noisy_result, n_error_paths = self.execute(
                    simulated_circuit, n_samples, gate_vecs, get_n_error_paths=True)
                sub_fs.append(hellinger_fidelity(
                    noisy_result, true_result))
                sub_eps.append(n_error_paths)

            fidelity = sum(sub_fs)/len(sub_fs)
            noisy_count = sum(sub_eps)/len(sub_eps)

            fidelity_list.append(fidelity)
            n_error_paths_list.append(noisy_count)

        return fidelity_list, n_error_paths_list

    @staticmethod
    def to_qiskit(circuit: Circuit) -> QuantumCircuit:
        """
        Convert a Circuit object to a Qiskit QuantumCircuit object.

        Args:
            circuit (Circuit): The quantum circuit to be converted.

        Returns:
            QuantumCircuit: The converted Qiskit QuantumCircuit object.
        """
        operated_qubits = circuit.operated_qubits

        new_circuit = circuit.copy()
        for gate in new_circuit.gates:
            gate['qubits'] = [operated_qubits.index(
                qubit) for qubit in gate.qubits]
        new_circuit.n_qubits = len(operated_qubits)

        qiskit_circuit = new_circuit.to_qiskit()
        # qiskit_circuit.measure_all()

        if circuit.measured_qubits is not None:
            new_creg = qiskit_circuit._create_creg(len(circuit.measured_qubits), "meas")
            qiskit_circuit.add_register(new_creg)
            qiskit_circuit.barrier()
            qiskit_circuit.measure([operated_qubits.index(qubit) for qubit in circuit.measured_qubits], new_creg)
        else:
            qiskit_circuit.measure_all()

        return qiskit_circuit

    def get_noise_model(self, qubit_mapping: list[int],):
        """
        Get the noise model for circuit execution.

        Args:
            qubit_mapping (list[int]): A mapping of qubits in the circuit to qubits on the backend.

        Returns:
            NoiseModel: The noise model for circuit execution.
        """
        noise_model = NoiseModel()
        
        if self.gate_error_model is not None:
            self.gate_error_model.configure_noise_model(noise_model, qubit_mapping)
        
        if self.readout_error_model is not None:
            self.readout_error_model.configure_noise_model(noise_model, qubit_mapping)

        return noise_model

    def execute(self, circuit: Circuit, n_samples=2000, gate_vecs: np.ndarray = None, get_n_error_paths = False) -> dict[str, int]:
        """
        Execute a quantum circuit with noise.

        Args:
            circuit (Circuit): The quantum circuit to be executed.
            n_samples (int, optional): Number of samples for circuit execution. Default is 2000.
            gate_vecs (np.ndarray, optional): Gate vectors for the circuit. Default is None.
            get_n_error_paths (bool, optional): Flag indicating whether to return the number of error paths. Default is False.

        Returns:
            dict[str, int]: Dictionary containing the counts of different measurement outcomes.
            int: Number of error paths encountered during execution (if get_n_error_paths is True).
        """
        circuit, n_error_paths = self._inject_context_error(circuit, gate_vecs)
        noise_model = self.get_noise_model(circuit.operated_qubits)
        result = execute(self.to_qiskit(circuit), self.qasm_simulator,
                         noise_model=noise_model, shots=n_samples,)
        if get_n_error_paths:
            return result.result().get_counts(), n_error_paths
        else:
            return result.result().get_counts()

    
    def execute_noise_free(self, circuit: Circuit, n_samples=2000) -> dict[str, int]:
        '''
        description: call qiskit simulator to execute without noise

        '''
        return execute(self.to_qiskit(circuit), self.qasm_simulator, shots=n_samples,).result().get_counts()




    def _inject_context_error(self, circuit: Circuit, gate_vecs: np.ndarray = None) -> tuple[Circuit, int]:
        '''
        description: inject spacial and tempory error when a gete has a path in error model
        '''
        if self.gate_error_model is None:
            return circuit, 0
        
        vec_model = self.vec_model
        if vec_model is None:
            assert len(self.gate_error_model.high_error_paths) == 0
            return circuit

        device2error_path_indeices = defaultdict(list)
        for path in self.gate_error_model.high_error_paths:
            for device in vec_model.device_to_pathtable:
                if vec_model.has_path(device, path):
                    device2error_path_indeices[device].append(
                        vec_model.path_index(device, path))

        for device in device2error_path_indeices:
            device2error_path_indeices[device] = np.array(
                device2error_path_indeices[device])

        if gate_vecs is None:
            gate_vecs = vec_model.vectorize(circuit)

        error_circuit = Circuit([], circuit.n_qubits)

        total_n_error_paths = 0

        for layer in circuit:
            error_layer = Layer([])
            for gate in layer:
                device = extract_device(gate)

                gate_vec = gate_vecs[gate.index]

                path_indices = np.argwhere(gate_vec > 0.0001)

                n_error_paths = len(np.intersect1d(
                    path_indices, device2error_path_indeices[device]))
                total_n_error_paths += n_error_paths

                if n_error_paths > 0:
                    for qubit in gate['qubits']:
                        error_layer.append(Gate({
                            'qubits': [qubit],
                            'params': [(random.random() / 10 - 1 / 20) * pi * n_error_paths],
                            'name': 'rx'
                        }))

            error_circuit.append(layer)
            if len(error_layer) > 0:
                error_circuit.append(error_layer)
                
        error_circuit._sort_gates()
        error_circuit._sort_qubits()

        return error_circuit, total_n_error_paths
