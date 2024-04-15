import logging
logging.basicConfig(level=logging.INFO)
import sys
sys.path.append('/home/JanusQ-main/')
import os

from janusq.analysis.fidelity_prediction import FidelityModel
from janusq.baselines.fidelity_prediction.rb_prediction import RBModel
from janusq.simulator.gate_error_model import GateErrorModel

from janusq.analysis.vectorization import RandomwalkModel

from janusq.data_objects.backend import GridBackend

from janusq.simulator.noisy_simulator import NoisySimulator
import random
from janusq.dataset import load_dataset
from janusq.tools.ray_func import map

from janusq.data_objects.circuit import Circuit, SeperatableCircuit

from janusq.tools.ray_func import map
import numpy as np
import ray

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

runtime_envs = {"working_dir": os.getcwd}
ray.init(log_to_driver=False)


import io
import pickle


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        renamed_module = module
        if module.startswith("data_objects"):# or module.startwith("analysis") or module.startwith("data_objects")
            renamed_module = "janusq." + module

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()



import pickle
with open("examples/dataset/fidelity_dataset_18q.pkl", "rb") as f:
    circuits: list[SeperatableCircuit] = renamed_load(f)   # TODO: 整理成circuits一个数组，fidelity一个数组的形式

print(len(circuits))


# circuits = circuits[:200]

sub_circuits, sub_fidelities = [], []
for circuit in circuits:
    for sub_cir in circuit.seperatable_circuits:
        sub_circuits.append(sub_cir)
        sub_fidelities.append(sub_cir.ground_truth_fidelity)

print(len(sub_circuits), len(sub_fidelities))


from sklearn.model_selection import train_test_split
train_cirucits, test_circuits, train_fidelities, test_fidelities = train_test_split(sub_circuits,  sub_fidelities, test_size=.2)

n_qubits = 18
n_steps = 1
n_walks = 20
backend = GridBackend(3, 6)

vec_model = RandomwalkModel(n_steps = n_steps, n_walks = n_walks, backend = backend)
vec_model.train(circuits, multi_process = True, remove_redundancy = False)

fidelity_model = FidelityModel(vec_model)
fidelity_model.train((train_cirucits, train_fidelities), multi_process = False)


fidelity_1q_rb = {0: 0.9994094148043156, 1: 0.9993508083886652, 2: 0.9993513578387458, 3: 0.9996978330672296, 4: 0.9997258463524775, 
                           5: 0.9993898065578337, 6: 0.9998335484697743, 7: 0.9997460044815009,  8: 0.9997219426985601, 9: 0.9992924485427597, 
                           10: 0.9994018918682177, 11: 0.9998410411794697, 12: 0.9994231683912435, 13: 0.9995938422219371, 14: 0.9947661045069707, 
                           15: 0.9997576786354693, 16: 0.9998387638441334,  17: 0.9996691783504945} 
fidelity_2q_rb = {(5,11): 0.993651602350742, (11,17): 0.9943374306798481,  (4,5): 0.9810612795342519,  (10,11): 0.9915544427978213,  
                           (16,17): 0.9908639448675425,  (4,10): 0.9914941121128581,  (10,16): 0.9868303060599511,  (3,4): 0.9899226069903224,  
                           (9,10): 0.9945250360193374,  (15,16): 0.9933864398113101,  (3,9): 0.991508018299962,  (9,15): 0.993773364368622,  
                           (2,3): 0.9802169505904027,  (8,9): 0.9912794178832776,  (14,15): 0.9867247971867894,  (2,8): 0.9765590682588615,  
                           (8,14): 0.9863913339619792,  (1,2): 0.9713229087974011,  (7,8): 0.9908463216114999,  (13,14): 0.9564265490465305,  
                           (1,7): 0.9856880460026779,  (7,13): 0.9935440562158602,  (0,1): 0.9833453296232256,  (6,7): 0.9939901490743566,  
                           (12,13): 0.9821366244436676,  (0,6): 0.9861987068804432,  (6,12): 0.9863008252688662} 


rb_fidelities = np.array(map(lambda circuit: RBModel.get_rb_fidelity(circuit, fidelity_1q_rb, fidelity_2q_rb), test_circuits))
janusct_fidelities = np.array(map(lambda circuit: fidelity_model.predict_circuit_fidelity(circuit), test_circuits))


from janusq.tools.plot import plot_scaater

durations = np.array([cir.duration for cir in test_circuits])

fig_quct, axes_quct = plot_scaater(test_fidelities, janusct_fidelities, durations, title = f"janusct inaccuracy = {np.abs(test_fidelities - janusct_fidelities).mean()}")
fig_rb, axes_rb = plot_scaater(test_fidelities, rb_fidelities, durations, title = f"rb inaccuracy = {np.abs(test_fidelities - rb_fidelities).mean()}")


