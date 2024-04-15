'''
    TODO: 目前没有收敛需要，找到问题
'''
import logging
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

from tools.ray_func import map
import random
from simulator.noisy_simulator import NoisySimulator
from data_objects.backend import Backend, LinearBackend, GridBackend, FullyConnectedBackend
from data_objects.random_circuit import random_seperatable_circuits
from analysis.vectorization import RandomwalkModel
from simulator.gate_error_model import GateErrorModel
from analysis.fidelity_prediction import FidelityModel
from data_objects.circuit import SeperatableCircuit
import ray
ray.init(log_to_driver=False)

# 把 circuit 和 vecs, paths 分离

n_qubits = 18

backend = LinearBackend(n_qubits, 1)

circuits = random_seperatable_circuits(
    backend, 1000, 5, [100, 150, 200], [.4], True)

random_walk_model = RandomwalkModel(n_steps=2, n_walks=20, backend=backend)
gate_vecs_per_circuit = random_walk_model.train(
    circuits, multi_process=True, remove_redundancy=False)


all_paths = random_walk_model.all_paths()
high_error_paths = random.choices(all_paths, k=50)

error_model = GateErrorModel.random_model(backend=backend, high_error_paths=high_error_paths)
error_model.vec_model = random_walk_model

simulator = NoisySimulator(backend=backend, gate_error_model = error_model)

# res = get_rb_errors(simulator,multi_process=True)
# print(res)



def obtain_subcircuit_fidelity(circuit: SeperatableCircuit):
    return simulator.obtain_seperable_circuit_fidelity(circuit)[0]

ground_truth_fidelities = map(
    obtain_subcircuit_fidelity, circuits, show_progress=True, multi_process=True)

# 摊平
flaten_circuits, flaten_fidelities = [], []
for circuit, subfidelities in zip(circuits, ground_truth_fidelities):
    for subcircuit, subfidelity in zip(circuit.seperatable_circuits, subfidelities):
        flaten_circuits.append(subcircuit)
        flaten_fidelities.append(subfidelity)
        
fidelity_model = FidelityModel(random_walk_model)
fidelity_model.train((flaten_circuits, flaten_fidelities))

logging.info(fidelity_model.predict_circuit_fidelity(circuits[10]))
