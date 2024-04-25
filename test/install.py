from janusq.data_objects.circuit import Circuit
from janusq.cloud_interface import submit, get_result
qc = Circuit([], n_qubits = 3)
qc.h(0, 0)
result = submit(circuit=qc, label= 'GHZ', shots= 3000, run_type='simulator', API_TOKEN='')
result = get_result(result['data']['result_id'], run_type='simulator', result_format='probs')
print(result)
