import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
sys.path.append('..')

import logging
logging.basicConfig(level=logging.ERROR)
from janusq.optimizations.readout_mitigation.fem import Mitigator
from qiskit.quantum_info.analysis import hellinger_fidelity
from janusq.optimizations.readout_mitigation.fem.tools import npformat_to_statuscnt
from time import time
from janusq.dataset import protocol_8,ghz_8qubit


benchmark_circuits_and_results = protocol_8
ghz_output = ghz_8qubit



samples = 10000
qubits = 8


mitigator = Mitigator(qubits, n_iters = 2)
scores = mitigator.init(benchmark_circuits_and_results, group_size = 2, partation_methods=[
                         'max-cut'],multi_process=False, draw_grouping = True)


n_qubits = 5

outout_ideal = {'1'*n_qubits:samples*0.5,'0'*n_qubits:samples*0.5}
t_qufem_1_new = time()
output_fem = mitigator.mitigate(ghz_output[1],[i for i in range(n_qubits)], cho = 1 )
t_qufem_2_new = time()
output_fem = npformat_to_statuscnt(output_fem)

print("Janus-FEM time: ",t_qufem_2_new-t_qufem_1_new)
print("Janus-FEM fidelity: ",hellinger_fidelity(outout_ideal,output_fem))

