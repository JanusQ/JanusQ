'''We test fidelity prediction on both simulators and real-world quantum hardware'''
import logging
import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

from analysis.fidelity_prediction import FidelityModel

from simulator.gate_error_model import GateErrorModel

from analysis.vectorization import RandomwalkModel

from objects.random_circuit import random_circuits, random_circuit
from objects.backend import Backend, LinearBackend, GridBackend, FullyConnectedBackend

from simulator.noisy_simulator import NoisySimulator
import random

from tools.ray_func import map

# 把 circuit 和 vecs, paths 分离

n_qubits = 5

backend = LinearBackend(n_qubits, 1)

circuits = random_circuits(backend, 50, [30, 50, 100], [.4], True)

random_walk_model = RandomwalkModel(n_steps = 2, n_walks = 20, backend = backend)
gate_vecs_per_circuit = random_walk_model.train(circuits, multi_process=False, remove_redundancy = False)


all_paths = random_walk_model.all_paths()
high_error_paths = random.choices(all_paths, k = 20)

error_model = GateErrorModel.random_model(backend=backend, high_error_paths=high_error_paths)
error_model.vec_model = random_walk_model

simulator = NoisySimulator(backend=backend, gate_error_model = error_model)


# res = get_rb_errors(simulator,multi_process=False)
# print(res)

ground_truth_fidelities = map(lambda circuit: simulator.obtain_circuit_fidelity(circuit)[0], circuits, show_progress=True, multi_process=False)

fidelity_model = FidelityModel(random_walk_model)
fidelity_model.train((circuits, ground_truth_fidelities))

logging.info(fidelity_model.predict_circuit_fidelity(circuits[10]))

# TODO: on real world quantum hardware

fidelity_model.plot_path_error(top_k=20)

