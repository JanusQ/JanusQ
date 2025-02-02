from .provider import Provider
from qiskit import QuantumCircuit
from .cloud_provider import CloudManager
from .cloud_provider import get_IBM_service
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import time

class CloudProvider(Provider):
    def __init__(self, cloud_manager: CloudManager, backend_name: str):
        super().__init__()
        self.cloud_manager = cloud_manager
        self.backend_name = backend_name
        self.backend = cloud_manager.service.backend(backend_name)
        # print(f"Backend: {self.backend_name}")
        self.pass_manager = generate_preset_pass_manager(
            backend=self.backend, optimization_level=1
            )
        # print("CloudProvider initialized")
        
    def get_counts(self, qc: QuantumCircuit, shots: int):
        shots = 512
        
        taskid = self.cloud_manager.submit_task((self.backend_name, shots), qc)
        
        while True:
            result = self.cloud_manager.get_counts(taskid)
            if result is not None:
                return result
            time.sleep(1)

        return None
    

    # from qiskit_aer import AerSimulator
    # from qiskit.circuit.library import RealAmplitudes
    # from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
    # from qiskit.quantum_info import SparsePauliOp
    # from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # from qiskit_ibm_runtime import Session
    # from qiskit_ibm_runtime import SamplerV2 as Sampler
    # from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    # # Bell Circuit
    # qc = QuantumCircuit(2)
    # qc.h(0)
    # qc.cx(0, 1)
    # qc.measure_all()

    # # Run the sampler job locally using FakeManilaV2
    # fake_manila = FakeManilaV2()
    # pm = generate_preset_pass_manager(backend=fake_manila, optimization_level=1)
    # isa_qc = pm.run(qc)
    # sampler = Sampler(backend=fake_manila)
    # result = sampler.run([isa_qc]).result()

    # # Run the sampler job locally using AerSimulator.
    # # Session syntax is supported but ignored.
    # aer_sim = AerSimulator()
    # pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    # isa_qc = pm.run(qc)
    # with Session(backend=aer_sim) as session:
    #     sampler = Sampler(session=session)
    #     result = sampler.run([isa_qc]).result()