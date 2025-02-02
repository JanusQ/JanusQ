from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2,FakeKyiv
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService

# Bell Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.h(1)
qc.cx(1, 0)
qc.measure_all()
 
# Run the sampler job locally using FakeManilaV2
fake_manila = FakeKyiv()
pm = generate_preset_pass_manager(backend=fake_manila, optimization_level=3)
isa_qc = pm.run(qc)
 
# You can use a fixed seed to get fixed results.
options = {"simulator": {"seed_simulator": 42}}
sampler = Sampler(mode=fake_manila, options=options)
 
result = sampler.run([isa_qc],shots=1000).result()
pub_result = result[0]
counts = pub_result.data.meas.get_counts()
print(counts)