from .provider import Provider, EXTENDED_BASIS_GATES
from qiskit import QuantumCircuit
from mqt import ddsim
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class DdsimProvider(Provider):
    def __init__(self):
        super().__init__()
        self.backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=2,
            basis_gates=EXTENDED_BASIS_GATES,
        )

    def get_counts(self, qc: QuantumCircuit, shots: int):
        job = self.backend.run(qc, shots=shots)
        counts = job.result().get_counts()
        return counts
        # counts = job.result().get_counts(qc)
