import logging
import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

from simulator.noisy_simulator import NoisySimulator
from data_objects.backend import LinearBackend
from simulator.readout_error_model import ReadoutErrorModel
from optimizations.readout_mitigation.fem import Mitigator
from optimizations.readout_mitigation.fem import IterativeSamplingProtocol, EnumeratedProtocol
import numpy as np
from data_objects.algorithms import get_algs, ibu_response_matrix
from qiskit.quantum_info.analysis import hellinger_fidelity
from optimizations.readout_mitigation.fem.tools import npformat_to_statuscnt,statuscnt_to_npformat

n_qubits = 5
samples = 1000
backend = LinearBackend(n_qubits, 1)
simulator = NoisySimulator(
    backend, readout_error_model=ReadoutErrorModel.random_model(backend))

circuit = get_algs(n_qubits, backend, algs = ['ghz'])[0]
output_noise = simulator.execute(circuit,samples)
outout_ideal = {'1'*n_qubits:samples*0.5,'0'*n_qubits:samples*0.5}



from baselines.readout_calibration.IBU.src.IBU import IBU
from baselines.readout_calibration.IBU.utils.qc_utils import *
from baselines.readout_calibration.IBU.utils.data_utils import *

matrices = []
for i in range(n_qubits):
    qc_t_0 = ibu_response_matrix(n_qubits, backend,i,0)[0]
    qc_t_1 = ibu_response_matrix(n_qubits, backend,i,1)[0]
    output_0 = simulator.execute(qc_t_0,1000)
    output_1 = simulator.execute(qc_t_1,1000)
    p_0_0 = output_0['0']/1000
    p_1_1 = output_1['1']/1000
    mat = np.array([[p_0_0, 1-p_0_0], [1-p_1_1, p_1_1]])
    matrices.append(mat)

params = {
    "exp_name": "ghz",
    "method": "reduced",  # options: "full", "reduced"
    "library": "jax",  # options: "tensorflow" (for "full" only) or "jax"
    "num_qubits": n_qubits,
    "max_iters": 100,
    "tol": 1e-4,
    "use_log": False,  # options: True or False
    "verbose": True,
    "init": "unif",  # options: "unif" or "unif_obs" or "obs"
    "smoothing": 1e-8,
    "ham_dist": 3
}


ibu = IBU(matrices, params)

ibu.set_obs(dict(output_noise))
ibu.initialize_guess()
t_sol, max_iters, tracker = ibu.train(params["max_iters"], tol=params["tol"], soln=outout_ideal)
outout_ibu = ibu.guess_as_dict()




protocol = EnumeratedProtocol(n_qubits)
real_bstrs, circuits_protocol = protocol.gen_circuits()

all_statuscnts = [
    simulator.execute(cir_protocol, samples)
    for cir_protocol in circuits_protocol
]


protocol_dataset = [statuscnt_to_npformat(p) for p in all_statuscnts]
protocol_dataset_all =  [[x, y] for x, y in zip(real_bstrs, protocol_dataset)]
protocol_iterative = IterativeSamplingProtocol(backend, hyper = 1, n_samples_iter = 2, threshold = 1e-7)
statuscnts = protocol_iterative.get_data(protocol_dataset_all)
protocol_statuscnts = [npformat_to_statuscnt(statuscnts[i][1]) for i in range(len(statuscnts))]
bstrs = [statuscnts[i][0] for i in range(len(statuscnts))]



mitigator = Mitigator(n_qubits, n_iters = 2)
scores = mitigator.init((bstrs, protocol_statuscnts), group_size=2, partation_methods=[
                         'max-cut'],multi_process=False)
output_fem = npformat_to_statuscnt(mitigator.mitigate(output_noise))


logging.info("Uncalibrated Algorithm Fidelity: ",hellinger_fidelity(outout_ideal,output_noise))
logging.info(hellinger_fidelity(outout_ideal,outout_ibu).item())
logging.info(hellinger_fidelity(outout_ideal,output_fem))
