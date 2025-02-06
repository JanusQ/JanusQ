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
        # shots = 512
        taskid = self.cloud_manager.submit_task((self.backend_name, shots), qc)
        
        while True:
            result = self.cloud_manager.get_counts(taskid)
            if result is not None:
                return result
            time.sleep(1)