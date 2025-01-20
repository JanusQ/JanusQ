from abc import ABC, abstractmethod
from typing import Dict, Union, Callable
from qiskit import QuantumCircuit
from qiskit.providers import Backend, BackendV2
from qiskit.transpiler import PassManager
import time

CORE_BASIS_GATES = ["measure", "cx", "id", "rz", "sx", "x"]
EXTENDED_BASIS_GATES = [
    "measure", "cx", "id", "s", "sdg", "x", "y", "h", "z", "mcx", 
    "cz", "sx", "sy", "t", "tdg", "swap", "rx", "ry", "rz",
]


class Provider(ABC):
    """ 请为每个solver创建单独的provider用于计时计数 """
    def __init__(self):
        self.backend: Union[Backend, BackendV2] = None
        self.pass_manager: PassManager = None
        # 量子电路执行时间
        self.quantum_circuit_execution_time = 0
        # 运行计数
        self.run_count = 0


    @abstractmethod
    def get_counts(self, qc: QuantumCircuit, shots: int) -> Dict:
        pass

    def get_counts_with_time(self, qc: QuantumCircuit, shots: int) -> Dict:
        start_time = time.perf_counter()  # 使用 perf_counter 记录开始时间
        result = self.get_counts(qc, shots)  # 调用子类实现的 get_counts 方法
        end_time = time.perf_counter()  # 使用 perf_counter 记录结束时间
        self.quantum_circuit_execution_time += end_time - start_time  # 计算耗时
        self.run_count += shots
        return result
    
    def transpile(self, qc: QuantumCircuit):
        return self.pass_manager.run(qc)


class CustomProvider(Provider):
    """没有预设的Provider 可自定义传入三参"""

    def __init__(
        self,
        backend: Union[Backend, BackendV2],
        pass_manager: PassManager,
        get_counts_fun: Callable[[QuantumCircuit], dict],
    ):
        self.backend = backend
        self.pass_manager = pass_manager
        self.get_counts = get_counts_fun
