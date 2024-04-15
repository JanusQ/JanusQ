# %% [markdown]
# import packages

# %%
import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

# import os
# os.chdir('..')

from analysis.fidelity_prediction import FidelityModel
from baselines.fidelity_prediction.rb_prediction import RBModel
from simulator.gate_error_model import GateErrorModel

from analysis.vectorization import RandomwalkModel

from data_objects.random_circuit import random_circuits, random_circuit
from data_objects.backend import Backend, LinearBackend, GridBackend, FullyConnectedBackend

from simulator.noisy_simulator import NoisySimulator
import random
from dataset import load_dataset
from tools.ray_func import map


# %% [markdown]
# model settings

# %%
n_qubits = 18
n_steps = 2
n_walks = 20
backend = GridBackend(3, 6)


# %% [markdown]
# load dataset with ground truth fidelity

# %%
from data_objects.circuit import SeperatableCircuit

dataset_id = '20230321'
circuits: list[SeperatableCircuit] = load_dataset(dataset_id)

# %% [markdown]
# train upstream model, turn a circuit to vectors using random walk

# %%
random_walk_model = RandomwalkModel(n_steps = n_steps, n_walks = n_walks, backend = backend)
gate_vecs_per_circuit = random_walk_model.train(circuits, True, remove_redundancy = False)

# %% [markdown]
# select interaction patterns randomly, simulate interaction between gates

# %%
flaten_circuits, flaten_fidelities = [], []
for circuit in circuits:
    for sub_cir in circuit.seperatable_circuits:
        flaten_circuits.append(sub_cir)
        flaten_fidelities.append(sub_cir.ground_truth_fidelity)

# %% [markdown]
# train fidelity prediction model

# %%
fidelity_model = FidelityModel(random_walk_model)
fidelity_model.train((flaten_circuits, flaten_fidelities))

# %% [markdown]
# predict on test dataset

# %% [markdown]
# compare with RB predict model


