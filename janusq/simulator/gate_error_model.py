import numpy as np
from janusq.analysis.vectorization import RandomwalkModel

from janusq.data_objects.backend import Backend
from janusq.tools.saver import dump, load
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

class GateErrorModel():
    def __init__(self, backend: Backend, fs_1q: np.ndarray, fs_2q: np.ndarray, T1s: np.ndarray, T2s: np.ndarray, vec_model: RandomwalkModel = None):
        self.n_qubits = backend.n_qubits
        self.backend = backend
        
        self.fs_1q = fs_1q
        self.fs_2q = fs_2q
        self.T1s = T1s
        self.T2s = T2s
        
        self.high_error_paths = set()
        
        self.readout_error_model = None
        self.vec_model: RandomwalkModel = vec_model

    @staticmethod
    def random_model(backend: Backend, fs_1q_range = [0.99, 1], fs_2q_range = [0.98, 0.995], T1_range = [1e8, 1e10], T2_range= [1e7, 1e8], high_error_paths = []):

        n_qubits = backend.n_qubits
        
        fs_1q = np.random.uniform(fs_1q_range[0], fs_1q_range[1], (n_qubits))
        fs_2q = np.random.uniform(fs_2q_range[1], fs_2q_range[1], (len(backend.coupling_map)))
        T1s = np.random.uniform(T1_range[0], T1_range[1], (n_qubits))
        T2s = np.random.uniform(T2_range[0], T2_range[1], (n_qubits))
        
        error_model = GateErrorModel(backend, fs_1q, fs_2q, T1s, T2s)
        error_model.high_error_paths = list(high_error_paths)
        
        return error_model
    
    @staticmethod
    def load(name):
        return load(name)

    def save(self, name):
        dump(name, self)
        
    def configure_noise_model(self, noise_model: NoiseModel, qubit_mapping: list[int]):
        '''
            qubit_mapping: e.g. [2, 3, 4] : this circuit only use 2, 3, 4, construct a noise model for these qubits
        '''

        backend = self.backend
        single_qubit_time = backend.single_qubit_gate_time
        two_qubit_time = backend.two_qubit_gate_time

        # construct noise model based on the new mapping
        for qubit in qubit_mapping:
            error = 1 - self.fs_1q[qubit]
            thermal_error = thermal_relaxation_error(self.T1s[qubit] * 1e3, self.T2s[qubit] * 1e3,
                                                    single_qubit_time)  # ns, ns ns

            total_qubit_error = thermal_error.compose(
                depolarizing_error(error, 1))
            noise_model.add_quantum_error(
                total_qubit_error, backend.basis_single_gates, [qubit_mapping.index(qubit)])

        for index, (qubit1, qubit2) in enumerate(backend.coupling_map):
            if qubit1 not in qubit_mapping or qubit2 not in qubit_mapping:
                continue

            error = 1 - self.fs_2q[index]

            thermal_error_q1 = thermal_relaxation_error(
                self.T1s[qubit1] * 1e3, self.T2s[qubit1] * 1e3, two_qubit_time)
            thermal_error_q2 = thermal_relaxation_error(
                self.T1s[qubit2] * 1e3, self.T2s[qubit1] * 1e3, two_qubit_time)
            thermal_error = thermal_error_q1.expand(thermal_error_q2)

            total_coupler_error = thermal_error.compose(
                depolarizing_error(error, 2))

            noise_model.add_quantum_error(
                total_coupler_error, backend.basis_two_gates, [qubit_mapping.index(qubit1), qubit_mapping.index(qubit2)])

        return noise_model