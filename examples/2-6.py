# %% [markdown]
# # Extend Janus-CT To Identify Bugs in the Quantum Circuit
# 
# **Author:** Congliang Lang \& Siwei Tan  
# 
# **Date:** 15/4/2024
# 
# Based on "[QuCT: A Framework for Analyzing Quantum Circuit by Extracting Contextual and Topological Features (MICRO 2023][1]"
# 
# [1]: https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/abs/10.1145/3613424.3614274%3Fcasa_token%3DffjIB1hQ4ZwAAAAA:8MajDLrDOC74WoeMf7r7AoQ-koxCa4E1TNqQg3GSDz03xUX6XdE3toNTM-YdM_e4rKEusMceJ6BGJg&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=11146218754516883150&ei=42YSZpPlFL6s6rQPtt6x6Ac&scisig=AFWwaeYaiu2hyx8HUJ_7Buf9Mwom
# 
# The vectorization of Janus-CT can be extended to more downstream tasks. For example, in this notebook, we use Janus-CT to identify the potential bugs in the quantum algorithm implementation. We apply a data driven method that traing a model to predict the error rate 

# %%
import sys
sys.path.append('/home/JanusQ-main/')
import logging
logging.basicConfig(level=logging.WARN)
# import ray
# ray.init(log_to_driver=False)

from janusq.data_objects.algorithms import get_algorithm_circuits
import random
import seaborn as sns
import numpy as np
from collections import defaultdict
import pandas as pd
import jax.numpy as jnp

from janusq.analysis.vectorization import RandomwalkModel

from janusq.tools.ray_func import map
from janusq.data_objects.backend import LinearBackend
import copy
import statistics
from janusq.data_objects.circuit import Circuit

# %%
from collections import Counter

class BugIdentificationModel:
    def __init__(self, vec_model: RandomwalkModel) -> None:
        self.vec_model = vec_model
    
    def train(self, algorithm_to_circuirts: dict[str, list[Circuit]]):
        self.total_vecs = []
        self.functionalities = []
        
        algorithm_names = list(algorithm_to_circuirts.keys())
        for algorithm_name, circuits in algorithm_to_circuirts.items():
            for circuit in circuits:
                vecs = self.vec_model.vectorize(circuit)
                self.total_vecs += list(vecs)
                self.functionalities += [algorithm_name] * len(vecs)
                # .index(algorithm_name)

        self.total_vecs = np.array(self.total_vecs)
        self.functionalities = np.array(self.functionalities)

    def identify_bug(self, circuit: Circuit, top_k = 3, dist_threshold = 1.6):
        gate_vecs = self.vec_model.vectorize(circuit)
        
        functionalities_per_gate = []
        all_functionalities = []
        for analyzed_vec in gate_vecs:
            dists = np.sqrt(np.sum((self.total_vecs - analyzed_vec)**2, axis=1))
            
            nearest_dist_indices = np.argsort(dists)[:top_k]
            nearest_dists = dists[nearest_dist_indices]
            
            print(nearest_dists[0])
            
            # nearest_dist_indices = nearest_dist_indices[nearest_dists < dist_threshold]
            # nearest_dist_indices = nearest_dist_indices[nearest_dists < dist_threshold]
            
            nearest_functionalities = self.functionalities[nearest_dist_indices]
            # nearest_functionalities = np.concatenate([self.functionalities[nearest_dist_indices], self.functionalities[np.where(dists == 0)[0]]])
        
            # nearest_functionalities = [algorithm_names[i]  for i in nearest_functionalities.tolist()]
            # nearest_functionalities = list(set([algorithm_names[i]  for i in nearest_functionalities.tolist()]))
            functionalities_per_gate.append(nearest_functionalities)
            all_functionalities += list(nearest_functionalities)
        
        top_functionalities = [
            functionality
            for functionality, count in Counter(all_functionalities).most_common(top_k)
            if count / circuit.n_gates > 0.2
        ]
        
        predicted_gate_indices = []
        for i, possible_functionalities in enumerate(functionalities_per_gate):
            if len([functionality for functionality in possible_functionalities if functionality in top_functionalities]) != 0:
                continue
            predicted_gate_indices.append(i)
        
        # print(circuit)
        return predicted_gate_indices



# %%
def construct_negatives(circuit: Circuit, n_error_gates, basis_gates):
    
    n_qubits = circuit.n_qubits
    bug_circuit = circuit.copy()
    for gate in bug_circuit.gates:
        gate.vec = None
    
    bug_start = random.randint(0, max(circuit.n_gates - 1 - n_error_gates, 1))
    bug_end = bug_start + n_error_gates
    bug_gate_ids = list(range(bug_start, min(bug_end, circuit.n_gates)))

    for bug_gate_id in bug_gate_ids:
        
        gate = bug_circuit.gates[bug_gate_id]

        name = random.choice(basis_gates) # ['rx', 'ry', 'rz', 'h', 'cz', 'cx']

        params = np.random.random((3,)) * 2 * np.pi
        params = params.tolist()
        
        qubit1 = random.randint(0, n_qubits - 1)
        qubit2 = random.choice([qubit for qubit in range(n_qubits) if qubit != qubit1])
        qubits = [qubit1, qubit2]
        
        gate['name'] = name
        if name in ('rx', 'ry', 'rz'):
            gate['qubits'] = qubits[:1]
            gate['params'] = params[:1]
            
        elif name in ('cz', 'cx'):
            gate['qubits'] = qubits
            gate['params'] = []
            
        elif name in ('h'):
            gate['qubits'] = qubits[:1]
            gate['params'] = []
            
        elif name in ('u'):
            gate['qubits'] = qubits[:1]
            gate['params'] = params
            
        else:
            logging.error("no such gate")
            return circuit

    bug_circuit.name = bug_circuit.name
    return bug_circuit, bug_gate_ids

# %%
algorithm_names = ['qft', 'hs', 'ising', 'qknn', 'qsvm', 'vqc', 'ghz', 'grover']
algorithm_to_circuirts = defaultdict(list)
algorithm_circuits = []

n_qubits = 3
backend = LinearBackend(n_qubits)
vec_model = RandomwalkModel(n_steps = 2, n_walks = 40, backend = backend, alpha=.5)

for n_qubits in range(n_qubits, backend.n_qubits + 1):
    for algorithm, circuit in zip(algorithm_names, get_algorithm_circuits(n_qubits, backend, algorithm_names)):
        algorithm_to_circuirts[algorithm].append(circuit)
        algorithm_circuits.append(circuit)

vec_model.train(algorithm_circuits)

# %%
bug_indentify_model = BugIdentificationModel(vec_model)
bug_indentify_model.train(algorithm_to_circuirts)

# %%
for circuit in algorithm_circuits:
    error_circuit, error_gate_indices = construct_negatives(circuit, n_error_gates=3, basis_gates= backend.basis_gates)
    
    # print(circuit)
    # print(error_circuit)
    predict_indices = bug_indentify_model.identify_bug(error_circuit)

    print(error_gate_indices, predict_indices)


# correct_rate = 0
# for predict_indice in predict_indices:
#     if predict_indice in bug_gate_ids:
#         correct_rate+=1

# print(str.format("identify_rate: {}",  correct_rate * 100 / len(bug_gate_ids)))

# %%



