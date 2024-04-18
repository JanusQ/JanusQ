"""Perform readout calibration on the GHZ circuit

The output with noise is entered into rapid readout calibration using ''mitigate()''

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
Mitigator: int,int
    qubits represents the number of qubit
    n_iters represents the number of iteration

qubits: int
    the number of qubit

ghz_output: list
    the output results of a noiseless GHZ circuit

benchmark_circuits_and_results: dict
    the results of running benchmark circuits

group_size : int
    qubit group size

Returns
-------
output_fem: dict
    the probability distribution after calibration
"""


import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
sys.path.append('..')
import logging
logging.basicConfig(level=logging.ERROR)
from janusq.optimizations.readout_mitigation.fem import Mitigator
from qiskit.quantum_info.analysis import hellinger_fidelity
from janusq.optimizations.readout_mitigation.fem.tools import npformat_to_statuscnt
from janusq.dataset import protocol_8,ghz_8qubit

benchmark_circuits_and_results = protocol_8
ghz_output = ghz_8qubit     


qubits = 8


mitigator = Mitigator(qubits, n_iters = 2)
scores = mitigator.init(benchmark_circuits_and_results, group_size = 2, partation_methods=[
                         'max-cut'],multi_process=False, draw_grouping = True)


n_qubits = 4
outout_ideal = {'1'*n_qubits:0.5,'0'*n_qubits:0.5}
output_fem = mitigator.mitigate(ghz_output[0],[i for i in range(n_qubits)], cho = 1 )
output_fem = npformat_to_statuscnt(output_fem)

print("Raw fidelity: ",hellinger_fidelity(outout_ideal,ghz_output[0]))
print("Janus-FEM fidelity: ",hellinger_fidelity(outout_ideal,output_fem))

