# %%

import logging
logging.basicConfig(level=logging.INFO)
import sys
sys.path.append('/home/JanusQ-main/')

from janusq.analysis.vectorization import RandomwalkModel, extract_device

from janusq.data_objects.random_circuit import random_circuits, random_circuit
from janusq.data_objects.backend import GridBackend

import random
import numpy as np


# %% [markdown]
# # Vectorization Model Of Janus-CT
# **Author:** Siwei Tan  
# 
# **Date:** 7/4/2024
# 
# Based on "[QuCT: A Framework for Analyzing Quantum Circuit by Extracting Contextual and Topological Features (MICRO 2023][1]"
# 
# [1]: https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/abs/10.1145/3613424.3614274%3Fcasa_token%3DffjIB1hQ4ZwAAAAA:8MajDLrDOC74WoeMf7r7AoQ-koxCa4E1TNqQg3GSDz03xUX6XdE3toNTM-YdM_e4rKEusMceJ6BGJg&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=11146218754516883150&ei=42YSZpPlFL6s6rQPtt6x6Ac&scisig=AFWwaeYaiu2hyx8HUJ_7Buf9Mwom

# %% [markdown]
# In the current Noisy Intermediate-Scale Quantum era, quantum circuit analysis is an essential technique for designing high-performance quantum programs. Current analysis methods exhibit either accuracy limitations or high computational complexity for obtaining precise results. To reduce this tradeoff, we propose Janus-CT, a unified framework for extracting, analyzing, and optimizing quantum circuits. The main innovation of Janus-CT is to vectorize each gate with each element, quantitatively describing the degree of the interaction with neighboring gates. Extending from the vectorization model, we can develope multiple downstream models for fidelity prediction and unitary decomposition, etc. In this tutorial, we introduce the APIs of the vectorization model of Janus-CT.

# %% [markdown]
# ## Vectorization Flow
# Below is the workflow to vectorize a gate in the quantum circuit. The gate is vectorized by two steps. The first step runs random walks to extract circuit features in the neighbor of the gates. the second step use a table comparison to generate the gate vector.
# 
# <div style="text-align:center;">
#     <img src="pictures/2-1.feature_extraction.png"  width="70%" height="70%">
# </div>

# %% [markdown]
# ## Random walk
# We apply random walk to extract the topological and contextual information of gates in the quantum circuit. Here is a example of random walk.

# %%
# generate a circuit
from janusq.analysis.vectorization import walk_from_gate

backend = GridBackend(2, 2)
circuit = random_circuit(backend, 10, .5, False)
print(circuit)

# choose a target gate
gate = random.choice(circuit.gates)

# apply random walk
paths = walk_from_gate(circuit, gate, 4, 2, backend.adjlist)

print('target gate:', gate)
print('generate paths:', paths)

# %% [markdown]
# The code generates 4 paths. Each path has at most 2 steps. A step is represented as "gate type,qubits-dependency-gate type,qubits". For example, "u,4-parallel-u,0-parallel-u,8" means that a U gate on qubit 4 is executed in parallel with U gates on qubits 0 and 8. 

# %% [markdown]
# ## Construction of Path Table
# 
# For a gate that requires vectorization, we compare it with a path table. The path table is off-line generated by applying random walks to a circuit dataset. To limits the size of the table, the table is usually hardware-specific.

# %%
# define the information of the quantum device
n_qubits = 6
backend = GridBackend(2, 3)

# generate a dataset including varous random circuits
circuit_dataset = random_circuits(backend, n_circuits=100, n_gate_list=[30, 50, 100], two_qubit_prob_list=[.4], reverse=True)

# apply random work to consturct the vectorization model with a path table
n_steps = 1
n_walks = 100
vec_model = RandomwalkModel(n_steps = n_steps, n_walks = n_walks, backend = backend, alpha= .5)
vec_model.train(circuit_dataset, multi_process=False)

print('length of the path table is', len(vec_model.pathtable))

# %% [markdown]
# # Vectorize a gate
# 
# As mentioned above, the vectorization of a gate is performed by comparing the generated paths with a path table. In JanusQ, we provide a api to do this. Below is a example of it.

# %%
# generate a circuit
circuit = random_circuit(backend, 10, .5, False)

# choose a target gate
gate = random.choice(circuit.gates)

# vectorization
vec = vec_model.vectorize(circuit, [gate])[0]
print('vector is', vec)

# %% [markdown]
# The indexes of the non-zero elements in the vector is same to the indexes of the generated paths in the path table, which is verified by following codes.

# %%
indexes = np.argwhere(vec > 0).flatten()
generated_paths = walk_from_gate(circuit, gate, 100, 1, backend.adjlist)
device = extract_device(gate)

print(list(indexes), '=', sorted([vec_model.path_index(device, path) for path in generated_paths]))

# %% [markdown]
# ## Reconstruction
# The vectorization of JanusQ-CT also allows the reconstruction of the sub-circuit around the gate by its vector.

# %%
# TODO: 检查下对不对
# checked

circuit = vec_model.reconstruct(device, vec)
print(circuit)

# %%



