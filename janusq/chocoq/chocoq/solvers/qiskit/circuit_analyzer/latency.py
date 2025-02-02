from typing import Iterable, Union, Tuple
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec, FakeAlmadenV2, FakeBelemV2, FakeSantiagoV2


class Latency:
    def __init__(self, backend=AerSimulator()) -> None:
        self.backend = backend
        # self.quired_dict = {}
    def calculate(self, gate: str, qubit: Union[int, Iterable[int]]):
        if gate not in self.backend.operation_names:
            print(gate)
            raise ValueError("the input gate is not in basis")
        # key = (gate, tuple(qubit))
        # if key in self.quired_dict:
        #     return self.quired_dict[key]
        if hasattr(self.backend, "_props_dict"):
            gate_list = self.backend._props_dict['gates']
            for gate_dict in gate_list:
                if gate_dict['gate']==gate:
                    # if gate_dict['qubits'] == qubit: # o.O?
                    if set(gate_dict['qubits']).intersection(set(qubit)) == set(qubit):
                        value = gate_dict['parameters'][1]['value']
                        # self.quired_dict[key] = value
                        return value
        else:
            return 0

## test code
if  __name__ =='__main__':
    print('test code')
    gate = 'ecr'
    qubit = [112, 108]
    print(f'{gate}: {Latency().calculate(gate, qubit)}')