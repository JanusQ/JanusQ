from .provider import Provider, CORE_BASIS_GATES, EXTENDED_BASIS_GATES
from qiskit import QuantumCircuit
from typing import Dict
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    phase_amplitude_damping_error,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
)
from typing import List, Union



def build_noise_model(
    p_meas: float,
    p_reset: float,
    p_gate1: float,
    t1: float,
    t2: float,
    depolar_param: float,
    damping_param: float,
    single_qubit_gates: List = ["id", "sx", "x"],
) -> NoiseModel:
    """
    Build a noise model for a simulator.
    Args:
        p_meas: measurement error probability
        p_reset: reset error probability
        p_gate1: 1-qubit gate error probability
        p_gate2: 2-qubit gate error probability
        t1: relaxation time of first qubit
        t2: relaxation time of second qubit
    """
    noise_model = NoiseModel()

    # 泡利噪声
    error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)
    noise_model.add_all_qubit_quantum_error(error_reset, "reset")
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cz"])

    # 退极化噪声
    # error_gate1 = depolarizing_error(depolar_param, 1)
    # error_gate2 = error_gate1.tensor(error_gate1)
    # # Add errors to noise model
    # noise_model.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    # noise_model.add_all_qubit_quantum_error(error_gate2, ["cz"])
    # param_amp, param_phase = damping_param, damping_param
    # error_gate1 = phase_amplitude_damping_error(param_amp, param_phase)
    # error_gate2 = error_gate1.tensor(error_gate1)
    # # Add errors to noise model
    # noise_model = NoiseModel()
    # noise_model.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    # noise_model.add_all_qubit_quantum_error(error_gate2, ["cz"])


    # 退相干噪声
    # time_g1 = 10
    # time_cz = 68
    # time_reset = 1560  # 1 microsecond
    # time_measure = 1560  # 1 microsecond
    # # QuantumError objects
    # error_reset = thermal_relaxation_error(t1, t2, time_reset)
    # error_measure = thermal_relaxation_error(t1, t2, time_measure)
    # error_cz = thermal_relaxation_error(t1, t2, time_cz).expand(
    #     thermal_relaxation_error(t1, t2, time_cz)
    # )
    # error_gate1 = thermal_relaxation_error(t1, t2, time_g1)
    # # Add errors to noise model
    # noise_model.add_all_qubit_quantum_error(error_gate1, single_qubit_gates)
    # noise_model.add_all_qubit_quantum_error(error_reset, "reset")
    # noise_model.add_all_qubit_quantum_error(error_measure, "measure")
    # noise_model.add_all_qubit_quantum_error(error_cz, ["cz"])
    



    return noise_model


class SimulatorProvider(Provider):
    def __init__(
        self,
        p_meas: float = 0.01,
        p_reset: float = 1e-3,
        p_gate1: float = 1e-3,
        t1: float = 114e3,
        t2: float = 96.54e3,
        depolar_param: float = 0.01,
        damping_param: float = 0.01,
        single_qubit_gates: List = ["id", "sx", "x"],
    ) -> None:
        """
        Initialize a fake simulator provider.
        Args:
            p_meas: measurement error probability
            p_reset: reset error probability
            p_gate1: 1-qubit gate error probability
            p_gate2: 2-qubit gate error probability
            t1: relaxation time T1
            t2: relaxation time T2
            depolar_param: depolarizing parameter
            damping_param: damping parameter
            single_qubit_gates: list of single qubit gates to include in noise model
        """
        super().__init__()
        self.noise_model = build_noise_model(
            p_meas,
            p_reset,
            p_gate1,
            t1,
            t2,
            depolar_param,
            damping_param,
            single_qubit_gates,
        )

    def get_counts(self, qc: QuantumCircuit, shots: int) -> Dict:
        # Create noisy simulator backend
        self.sim_noise = AerSimulator(noise_model=self.noise_model, shots=shots)
        # Transpile circuit for noisy basis gates
        circ_tnoise = transpile(
            qc,
            self.sim_noise,
            basis_gates=["id", "sx", "x", "rz", "cz", "reset", "measure"],
        )

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
        circ_tnoise = transpile(
            qc,
            self.sim_noise,
            basis_gates=["id", "sx", "x", "rz", "cz", "reset", "measure"],
        )
        return circ_tnoise
