from .provider import Provider
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeTorino, FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler


class FakeKyivProvider(Provider):
    def __init__(self):
        super().__init__()
        self.backend = FakeKyiv()
        self.pass_manager = generate_preset_pass_manager(
            backend=self.backend, optimization_level=2
        )

    def get_counts(self, qc: QuantumCircuit, shots: int):
        sampler = Sampler(mode=self.backend)
        job = sampler.run([qc], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
        return counts


class FakeTorinoProvider(Provider):
    def __init__(self):
        super().__init__()
        self.backend = FakeTorino()
        self.pass_manager = generate_preset_pass_manager(
            backend=self.backend, optimization_level=2
        )

    def get_counts(self, qc: QuantumCircuit, shots: int):
        sampler = Sampler(mode=self.backend)
        job = sampler.run([qc], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
        return counts


class FakeBrisbaneProvider(Provider):
    def __init__(self):
        super().__init__()
        self.backend = FakeBrisbane()
        self.pass_manager = generate_preset_pass_manager(
            backend=self.backend, optimization_level=2
        )

    def get_counts(self, qc: QuantumCircuit, shots: int):
        sampler = Sampler(mode=self.backend)
        job = sampler.run([qc], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
        return counts
