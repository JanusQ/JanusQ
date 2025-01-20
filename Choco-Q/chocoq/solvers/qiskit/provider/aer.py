from .provider import Provider, CORE_BASIS_GATES, EXTENDED_BASIS_GATES
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from typing import Dict

options = {"simulator": {"seed_simulator": 77}}


class AerProvider(Provider):
    def __init__(self):
        super().__init__()
        self.backend = AerSimulator()
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=2, basis_gates=CORE_BASIS_GATES,
        )

    def get_counts(self, qc: QuantumCircuit, shots: int) -> Dict:
        sampler = Sampler(mode=self.backend, options=options)
        job = sampler.run([qc], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
        return counts


class AerGpuProvider(Provider):
    def __init__(self):
        super().__init__()
        self.backend = AerSimulator(method="statevector", device="GPU")
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=3,
            basis_gates=EXTENDED_BASIS_GATES,
        )

    def get_counts(self, qc: QuantumCircuit, shots: int) -> Dict:
        sampler = Sampler(mode=self.backend, options=options)
        job = sampler.run([qc], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
        return counts
