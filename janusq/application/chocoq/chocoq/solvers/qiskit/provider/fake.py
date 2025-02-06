from .provider import Provider
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeTorino, FakeBrisbane,FakePeekskill
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

class FakeProvider(Provider):
    def __init__(self, fake_backend):
        super().__init__()
        self.backend = fake_backend()
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


class FakeKyivProvider(FakeProvider):
    def __init__(self):
        super().__init__(FakeKyiv)
    

class FakePeekskillProvider(FakeProvider):
    def __init__(self):
        super().__init__(FakePeekskill)


class FakeTorinoProvider(FakeProvider):
    def __init__(self):
        super().__init__(FakeTorino)


class FakeBrisbaneProvider(FakeProvider):
    def __init__(self):
        super().__init__(FakeBrisbane)