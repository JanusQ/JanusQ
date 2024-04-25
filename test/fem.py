from janusq.optimizations.readout_mitigation.fem import Mitigator
from qiskit.quantum_info.analysis import hellinger_fidelity
from janusq.optimizations.readout_mitigation.fem.tools import npformat_to_statuscnt
from janusq.dataset import protocol_8 as benchmark_circuits_and_results, ghz_8qubit as ghz_output

n_qubits = 8
mitigator = Mitigator(n_qubits, n_iters = 2)
scores = mitigator.init(benchmark_circuits_and_results, group_size = 2, partation_methods=['max-cut'],multi_process=False, draw_grouping = True)

qubits = 4
output_ideal = {'1'*4:0.5,'0'*4:0.5}
output_fem = mitigator.mitigate(ghz_output[0],measured_qubits = [i for i in range(qubits)], cho = 1 )
output_fem = npformat_to_statuscnt(output_fem)

print("Janus-FEM fidelity: ",hellinger_fidelity(output_ideal,output_fem))