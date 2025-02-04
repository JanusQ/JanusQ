import logging
import sys
import os
from pathlib import Path

sys.path.append(str(Path(os.getcwd())))

# this can be integrated as a pass of qiskit
from tools.ray_func import map
import random
from simulator.noisy_simulator import NoisySimulator
from objects.backend import Backend, LinearBackend, GridBackend, FullyConnectedBackend
from objects.random_circuit import random_circuits, random_circuit
from analysis.vectorization import RandomwalkModel
from simulator.gate_error_model import GateErrorModel
from analysis.fidelity_prediction import FidelityModel
from optimization.scheduling.scheduling_ct import Scheduler
from optimization.mapping.mapping_ct import Mapper
from analysis.vectorization import RandomwalkModel, extract_device
from collections import defaultdict

n_qubits = 5

backend = GridBackend(n_qubits, 1)

circuits = random_circuits(backend, 1000, range(30, 130, 10), [.1, .2, .3, .4, .5], True)

random_walk_model = RandomwalkModel(n_steps=1, n_walks=20, backend=backend)
gate_vecs_per_circuit = random_walk_model.train(
    circuits, multi_process=False, remove_redundancy=False)

all_paths = random_walk_model.all_paths()
high_error_paths = random.choices(all_paths, k=20)

error_model = GateErrorModel.random_model(
    backend=backend, high_error_paths=high_error_paths)
error_model.vec_model = random_walk_model

simulator = NoisySimulator(backend=backend, gate_error_model = error_model)
ground_truth_fidelities = map(lambda circuit: simulator.obtain_circuit_fidelity(circuit)[0], circuits, show_progress=True, multi_process=True)

fidelity_model = FidelityModel(random_walk_model)
fidelity_model.train((circuits, ground_truth_fidelities))

scheduler = Scheduler(fidelity_model)
mapper = Mapper(fidelity_model)

benchmarking_circuits = random_circuits(backend, 10, [50, 70], [.2, .4], True)

for circuit in benchmarking_circuits:
    opt_circuit = scheduler.run(circuit, timeout=30)
    
    before_opt_fidelity = simulator.obtain_circuit_fidelity(circuit)[0]
    after_opt_fidelity = simulator.obtain_circuit_fidelity(opt_circuit)[0]
    logging.info('before scheduling: ', before_opt_fidelity, 'after scheduling: ', after_opt_fidelity)

    mapping_backend = LinearBackend(n_qubits)
    
    opt_circuit, candicates = mapper.run(circuit, mapping_backend, return_candidates=True)
    
    before_opt_fidelity = [
        simulator.obtain_circuit_fidelity(circuit)[0]
        for circuit in candicates
    ]
    before_opt_fidelity = sum(before_opt_fidelity)/len(before_opt_fidelity)
    after_opt_fidelity = simulator.obtain_circuit_fidelity(opt_circuit)[0]
    
    logging.info('before mapping: ', before_opt_fidelity, 'after mapping: ', after_opt_fidelity)
    
