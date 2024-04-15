

from data_objects.backend import Backend
import numpy as np

from tools.saver import load, dump
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError

class ReadoutErrorModel():
    def __init__(self, backend: Backend, M_per_qubit: list[np.ndarray]):
        self.n_qubits = backend.n_qubits
        self.backend = backend
        
        self.M_per_qubit = M_per_qubit

    @staticmethod
    def random_model(backend: Backend):
        n_qubits = backend.n_qubits
        measure_fids = np.random.random(size=(n_qubits, 2))
        measure_fids = np.abs(measure_fids) / 10 + .9

        meas_mats = []
        for qubit in range(n_qubits):
            meas_mat = np.array([
                [measure_fids[qubit][0],    1-measure_fids[qubit][0]],
                [1-measure_fids[qubit][1],  measure_fids[qubit][1]]
            ])

            meas_mats.append(meas_mat)
            
        return ReadoutErrorModel(backend, meas_mats)
    
    @staticmethod
    def load(name):
        return load(name)

    def save(self, name):
        dump(name, self)
        
    def configure_noise_model(self, noise_model: NoiseModel, qubit_mapping: list[int]):
        '''
            qubit_mapping: e.g. [2, 3, 4] : this circuit only use 2, 3, 4, construct a noise model for these qubits
        '''
        
        for index, qubit in enumerate(qubit_mapping):
            re = ReadoutError(self.M_per_qubit[qubit])
            noise_model.add_readout_error(re, qubits=[index])
            
        return noise_model
        