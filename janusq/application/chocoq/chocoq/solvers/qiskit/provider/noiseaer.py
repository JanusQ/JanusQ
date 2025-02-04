from .provider import Provider, CORE_BASIS_GATES, EXTENDED_BASIS_GATES
from qiskit import QuantumCircuit
from typing import Dict
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,phase_amplitude_damping_error,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from typing import List, Union



def build_Pauli_noise_model(p_meas: float,p_reset: float, p_gate1: float, single_qubit_gates: List = ['id','sx','x']) -> NoiseModel:
    """
    Build a noise model for a simulator.
    Args:
        p_meas: measurement error probability
        p_reset: reset error probability
        p_gate1: 1-qubit gate error probability
        p_gate2: 2-qubit gate error probability
    """

    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)
    # error_gate2 = pauli_error()

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
    
    return noise_bit_flip 
        
def build_thermal_noise_model(t1: float, t2: float,  single_qubit_gates: List = ['id','sx','x']) -> NoiseModel:
    """
    Build a thermal noise model for a simulator.
    Args:
        t1: relaxation time of first qubit
        t2: relaxation time of second qubit
        gate_time: gate time
    """
   # QuantumError objects
   # Instruction times (in nanoseconds)
    time_g1 = 10
    time_cz = 68
    time_reset =  1560  # 1 microsecond
    time_measure =  1560 # 1 microsecond
    # QuantumError objects
    error_reset = thermal_relaxation_error(t1, t2, time_reset)
    error_measure = thermal_relaxation_error(t1, t2, time_measure)
    error_cz = thermal_relaxation_error(t1, t2,time_cz).expand(thermal_relaxation_error(t1, t2, time_cz))
    error_gate1 = thermal_relaxation_error(t1, t2, time_g1)
    # Add errors to noise model
    noise_thermal = NoiseModel()
    noise_thermal.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    noise_thermal.add_all_qubit_quantum_error(error_reset, "reset")
    noise_thermal.add_all_qubit_quantum_error(error_measure, "measure")
    noise_thermal.add_all_qubit_quantum_error(error_cz, ["cz"])
    return noise_thermal 

def build_depolarizing_noise_model(param: float, single_qubit_gates: List = ['id','sx','x','rz']) -> NoiseModel:
    """
    Build a depolarizing noise model for a simulator.
    Args:
        param: depolarizing parameter
        gate_time: gate time
    """
    # QuantumError objects
    error_gate1 = depolarizing_error(param, 1)
    error_gate2 = error_gate1.tensor(error_gate1)
    # Add errors to noise model
    noise_depolarizing = NoiseModel()
    noise_depolarizing.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    noise_depolarizing.add_all_qubit_quantum_error(error_gate2, ["cz"])
    return noise_depolarizing

def build_phase_amplitude_damping_error_model(gamma: float, single_qubit_gates: List = ['id','sx','x','rz']) -> NoiseModel:
    param_amp, param_phase = gamma, gamma
    error_gate1 = phase_amplitude_damping_error(param_amp, param_phase)
    error_gate2 = error_gate1.tensor(error_gate1)
    # Add errors to noise model
    noise_phase_amplitude_damping = NoiseModel()
    noise_phase_amplitude_damping.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    noise_phase_amplitude_damping.add_all_qubit_quantum_error(error_gate2, ["cz"])
    return noise_phase_amplitude_damping
        

class NoiseAerProvider(Provider):
    def __init__(self,**kwargs):
        super().__init__()
        
    def get_counts(self, qc: QuantumCircuit, shots: int) -> Dict:
        # Create noisy simulator backend
        self.sim_noise = AerSimulator(noise_model=self.noise_model, shots=shots)
        # Transpile circuit for noisy basis gates
        circ_tnoise = transpile(qc, self.sim_noise)

        # Run and get counts
        result_bit_flip = self.sim_noise.run(circ_tnoise).result()
        counts_bit_flip = result_bit_flip.get_counts(0)
        return counts_bit_flip
    
    def get_probabilities(self, qc: QuantumCircuit, shots: int) -> Dict:
        
        counts = self.get_counts(qc, shots)
        probabilities = {}
        for key, value in counts.items():
            probabilities[key] = value / shots
        return probabilities


    def transpile(self, qc: QuantumCircuit) -> QuantumCircuit:
        self.sim_noise = AerSimulator(noise_model=self.noise_model)
        # Transpile circuit for noisy basis gates
        circ_tnoise = transpile(qc, self.sim_noise)
        return circ_tnoise
    
class ThermalNoiseAerProvider(NoiseAerProvider):
    def __init__(self, t1: float, t2: float, **kwargs):
        super().__init__()
        self.noise_model = build_thermal_noise_model(t1, t2)

class BitFlipNoiseAerProvider(NoiseAerProvider):
    def __init__(self, p_meas: float, p_reset: float, p_gate1: float, **kwargs):
        super().__init__()
        self.noise_model = build_Pauli_noise_model(p_meas, p_reset, p_gate1)

class DepolarizingNoiseAerProvider(NoiseAerProvider):
    def __init__(self, param: float, **kwargs):
        super().__init__()
        self.noise_model = build_depolarizing_noise_model(param)

class PhaseAmplitudeDampingNoiseAerProvider(NoiseAerProvider):
    def __init__(self, gamma: float, **kwargs):
        super().__init__()
        self.noise_model = build_phase_amplitude_damping_error_model(gamma)



